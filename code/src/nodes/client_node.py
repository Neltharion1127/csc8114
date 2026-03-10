import grpc
import io
import torch
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.models.split_lstm import ClientLSTM
import time
import os
import json

# Use shared common module (provides `cfg` and `project_root`)
from src.shared.common import cfg, project_root
from src.shared.compression import compress, decompress

# Feature columns (can be overridden in config.yaml under data.feature_cols)
FEATURE_COLS = cfg.get("data", {}).get(
    "feature_cols",
    ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"],
)

# The date that separates training data from test data
SPLIT_DATE = pd.Timestamp("2026-02-28 00:00:00")


# --- Data Loading & Preprocessing ---

def _load_sensor_data(file_path: str) -> pd.DataFrame:
    """
    Reads a single sensor parquet file, ensures a DatetimeIndex,
    and computes the 'future_3h_rain' target column.
    Returns the preprocessed DataFrame.
    """
    df = pd.read_parquet(file_path)
    if 'Timestamp' in df.columns:
        df.set_index('Timestamp', inplace=True)
        df.index = pd.to_datetime(df.index)

    # Target: total rainfall over next 3 hours
    df['future_3h_rain'] = df['Rain'].shift(-3).rolling(window=3).sum()
    return df


# --- Balanced Training Sample Selection ---

def _sample_training_index(df: pd.DataFrame) -> tuple[int, str] | None:
    """
    Picks a random training index using balanced rain/dry sampling.
    Returns (target_idx, mode) or None if no valid samples exist.
    
    Strategy: 50% chance to force-pick a rainy sample to counter class imbalance.
    Only uses data before SPLIT_DATE, with enough history (>=24 rows) and future (>=3 rows).
    """
    split_pos = df.index.get_indexer([SPLIT_DATE], method='pad')[0]
    all_indices = np.arange(len(df))

    # Only train on data before the test cutoff, with valid window bounds
    base_mask = (
        (all_indices < split_pos) &
        (all_indices >= 24) &
        (all_indices < len(df) - 3)
    )

    rainy_pos = all_indices[base_mask & (df['future_3h_rain'] > 0)]
    dry_pos   = all_indices[base_mask & (df['future_3h_rain'] == 0)]

    if len(rainy_pos) > 0 and np.random.rand() > 0.5:
        return int(np.random.choice(rainy_pos)), "RAIN_SAMPLE"
    elif len(dry_pos) > 0:
        return int(np.random.choice(dry_pos)), "DRY_SAMPLE"
    return None  # No valid data in this sensor file


# --- Single-Step Forward & Backward Training ---

def _forward_and_backprop(
    stub,
    client_id: int,
    client_model: ClientLSTM,
    optimizer: torch.optim.Optimizer,
    df: pd.DataFrame,
    target_idx: int,
    target_value: float,
    mode: str,
    sensor_id: str,
    compression_mode: str,
) -> dict:
    """
    Runs one full split-learning step for a single training sample:
      1. Extract features from df
      2. Forward pass through ClientLSTM -> smashed activation
      3. Compress & send to Server via gRPC Forward()
      4. Receive gradient from Server
      5. Decompress & run backward pass
      6. Update optimizer

    Returns a log entry dict with loss, latency, and payload info.
    """
    # Extract input features: last 24 hours before target_idx
    raw_data = torch.tensor(
        df[FEATURE_COLS].iloc[target_idx - 24:target_idx].values,
        dtype=torch.float32
    )
    input_tensor = raw_data.unsqueeze(0)  # -> (1, 24, num_features)

    # Forward pass through client-side model
    smashed_activation = client_model(input_tensor)

    # Compress activations and build gRPC request
    start_time = time.time()
    activation_bytes = compress(smashed_activation, compression_mode)
    payload_size = len(activation_bytes)

    request = fsl_pb2.ForwardRequest(
        client_id=client_id,
        activation_data=activation_bytes,
        true_target=target_value,
        latency_ms=0.0,
        compression_mode=compression_mode,
    )
    print(f"[CLIENT] Transmitting activations for {sensor_id}... Payload: {payload_size} bytes")

    # Send to server, receive gradient feedback
    response = stub.Forward(request)
    latency_ms = (time.time() - start_time) * 1000.0

    # Parse loss from server's status message, e.g. "Success: Loss 0.0040"
    try:
        current_loss = float(response.status_message.split("Loss")[-1].split()[0].strip())
    except Exception:
        current_loss = 0.0

    # Display training progress
    icon = "💧💧💧" if target_value > 0 else "☁️"
    print(f"{icon} [{mode}] Sensor: {sensor_id[:10]} | 3h Target: {target_value:.2f} | Loss: {current_loss:.6f}")

    # Decompress gradient and run backward pass
    received_grad = decompress(response.gradient_data, smashed_activation.shape, compression_mode)
    smashed_activation.backward(received_grad)
    optimizer.step()

    print(f"[SERVER] Feedback processed for {sensor_id} | {response.status_message} | Latency: {latency_ms:.2f} ms")

    return {
        "Target": target_value,
        "RainFlag": 1 if target_value > 0 else 0,
        "Loss": current_loss,
        "LatencyMs": float(latency_ms),
        "PayloadBytes": payload_size,
    }


# --- End-of-Epoch FedAvg Synchronization ---

def _fed_avg_sync(stub, client_id: int, client_model: ClientLSTM) -> ClientLSTM:
    """
    Serializes the local ClientLSTM state dict, sends it to the server
    via Synchronize(), and loads the returned aggregated global weights.
    Returns the updated model.
    """
    buffer = io.BytesIO()
    torch.save(client_model.state_dict(), buffer)
    client_weights_bytes = buffer.getvalue()

    sync_req = fsl_pb2.SyncRequest(
        client_id=client_id,
        client_weights=client_weights_bytes
    )

    print(f"[CLIENT {client_id}] Waiting for global aggregation...")
    sync_res = stub.Synchronize(sync_req)

    if sync_res.global_weights:
        global_buffer = io.BytesIO(sync_res.global_weights)
        global_state_dict = torch.load(global_buffer, weights_only=True, map_location='cpu')
        client_model.load_state_dict(global_state_dict)
        print(f"[CLIENT {client_id}] Successfully loaded Global Model Round {sync_res.round_number}")
    else:
        print(f"[CLIENT {client_id}] Failed to get global weights.")

    return client_model


# --- Save Training Logs ---

def _save_results(client_id: int, experimental_logs: list) -> None:
    """
    Saves training logs as a CSV and a JSON metadata sidecar to results/.
    """
    output_dir = os.path.join(project_root, "results")
    os.makedirs(output_dir, exist_ok=True)

    log_df = pd.DataFrame(experimental_logs)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"training_log_client{client_id}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    log_df.to_csv(filepath, index=False)
    print(f"[CLIENT] Saved training log to {filepath}")

    meta = {
        "timestamp": timestamp,
        "csv": filename,
        "num_records": len(experimental_logs),
        "cfg": cfg,
    }
    meta_path = filepath.replace('.csv', '_meta.json')
    try:
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)
        print(f"[CLIENT] Saved metadata to {meta_path}")
    except Exception as e:
        print(f"[CLIENT WARN] Failed to write metadata: {e}")


# --- Main Orchestrator ---

def run_all_client(data_dir: str = "dataset/processed", epochs: int = 10) -> None:
    """
    Top-level entry point for a federated split learning client.

    Flow:
      1. Connect to server & register to get an assigned client_id
      2. Load the subset of sensor files allocated to this client
      3. For each epoch:
           a. For each sensor: load data, sample, forward/backprop
           b. Synchronize model weights with server (FedAvg)
      4. Save training logs
    """
    server_host = cfg.get("grpc", {}).get("server_host", "fsl-server")
    server_port = cfg.get("grpc", {}).get("server_port", 50051)
    target_address = f"{server_host}:{server_port}"
    compression_mode = cfg.get("compression", {}).get("mode", "float32")
    epochs = cfg.get("training", {}).get("num_rounds", epochs)

    time.sleep(cfg.get("training", {}).get("start_delay", 8))
    print(f"[CLIENT] Connecting to {target_address} for registration...")

    with grpc.insecure_channel(target_address) as channel:
        stub = fsl_pb2_grpc.FSLServiceStub(channel)

        # Step 1: Register with server to get an assigned ID
        reg = stub.Register(fsl_pb2.RegisterRequest())
        client_id, num_clients = reg.client_id, reg.total_clients
        print(f"[CLIENT] Registered — ID: {client_id} / {num_clients}")

        # Step 2: Partition sensor files for this client
        all_files = sorted(glob.glob(os.path.join(project_root, data_dir, "*.parquet")))
        chunk_size = len(all_files) // num_clients
        start_idx  = (client_id - 1) * chunk_size
        end_idx    = start_idx + chunk_size if client_id < num_clients else len(all_files)
        client_files = all_files[start_idx:end_idx]
        print(f"[CLIENT {client_id}] Allocated {len(client_files)}/{len(all_files)} sensors")

        # Step 3: Initialize model and training state
        client_model = ClientLSTM(
            input_size=cfg.get("model", {}).get("input_size", len(FEATURE_COLS)),
            hidden_size=cfg.get("model", {}).get("hidden_size", 64),
        )
        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=cfg.get("training", {}).get("lr", 0.001)
        )
        client_model.train()
        experimental_logs = []

        # Step 4: Training loop
        for epoch in range(epochs):
            print(f"[EPOCH {epoch+1}/{epochs}] Client {client_id} starting...")

            for file_path in client_files:
                optimizer.zero_grad()
                sensor_id = Path(file_path).stem

                try:
                    df = _load_sensor_data(file_path)
                    result = _sample_training_index(df)

                    if result is None:
                        experimental_logs.append({
                            "Epoch": epoch + 1, "Sensor": sensor_id,
                            "Target": -1, "RainFlag": -1,
                            "Loss": 0.0, "LatencyMs": 0.0,
                            "PayloadBytes": 0, "Status": "SKIPPED_NO_DATA",
                        })
                        continue

                    target_idx, mode = result
                    target_value = float(df['future_3h_rain'].iloc[target_idx])

                    log_entry = _forward_and_backprop(
                        stub, client_id, client_model, optimizer,
                        df, target_idx, target_value, mode, sensor_id, compression_mode,
                    )
                    experimental_logs.append({"Epoch": epoch + 1, "Sensor": sensor_id, **log_entry})

                except Exception as e:
                    print(f"[CLIENT {client_id} ERROR] {sensor_id}: {e}")

            # End-of-epoch: sync weights with server
            print(f"[CLIENT {client_id}] Epoch {epoch+1} done. Synchronizing...")
            try:
                client_model = _fed_avg_sync(stub, client_id, client_model)
            except Exception as e:
                print(f"[CLIENT {client_id}] Sync failed: {e}")

    # Step 5: Save results
    _save_results(client_id, experimental_logs)
    print(f"\n========== Training complete for Client {client_id}. ==========")


if __name__ == '__main__':
    run_all_client()
