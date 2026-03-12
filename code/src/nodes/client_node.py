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
from datetime import datetime

# Use shared common module (provides `cfg` and `project_root`)
from src.shared.common import cfg, project_root
from src.shared.compression import compress, decompress

# Feature columns (can be overridden in config.yaml under data.feature_cols)
FEATURE_COLS = cfg.get("data", {}).get(
    "feature_cols",
    ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"],
)

# The date that separates training data from test data (dynamic window for testing)
end_date_str = cfg.get("data_download", {}).get("end_date", "2026-03-10T00:00:00")
test_days = cfg.get("data", {}).get("test_days", 14)
SPLIT_DATE = pd.Timestamp(end_date_str).tz_localize(None) - pd.Timedelta(days=test_days)


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

def _sample_index(df: pd.DataFrame, is_training: bool = True) -> tuple[int, str] | None:
    """
    Picks a random index using balanced rain/dry sampling.
    Returns (target_idx, mode) or None if no valid samples exist.
    """
    try:
        split_pos = df.index.get_indexer([SPLIT_DATE], method='pad')[0]
    except Exception:
        split_pos = int(len(df) * 0.8) # fallback to 80/20 if date missing
        
    all_indices = np.arange(len(df))

    base_mask = (all_indices >= 24) & (all_indices < len(df) - 3)
    if is_training:
        base_mask = base_mask & (all_indices < split_pos)
    else:
        base_mask = base_mask & (all_indices >= split_pos)

    rainy_pos = all_indices[base_mask & (df['future_3h_rain'] > 0)]
    dry_pos   = all_indices[base_mask & (df['future_3h_rain'] == 0)]

    if len(rainy_pos) > 0 and np.random.rand() > 0.5:
        return int(np.random.choice(rainy_pos)), "RAIN_SAMPLE"
    elif len(dry_pos) > 0:
        return int(np.random.choice(dry_pos)), "DRY_SAMPLE"
    return None  # No valid data in this sensor file


# --- Single-Step Forward & Backward Training ---

def _forward_step(
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
    feat_stats: tuple[np.ndarray, np.ndarray] | None = None,
    is_training: bool = True,
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
    use_gpu = cfg.get("training", {}).get("use_gpu", False)
    device = torch.device("mps") if (use_gpu and torch.backends.mps.is_available()) else torch.device("cpu")
    
    raw_data = df[FEATURE_COLS].iloc[target_idx - 24:target_idx].values
    if feat_stats:
        mean, std = feat_stats
        raw_data = (raw_data - mean) / std
        
    raw_data_tensor = torch.tensor(
        raw_data,
        dtype=torch.float32,
        device=device
    )
    input_tensor = raw_data_tensor.unsqueeze(0)  # -> (1, 24, num_features)

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
        latency_ms=0.0, # TODO: implement true latency reporting
        compression_mode=compression_mode,
        is_training=is_training
    )
    phase = "TRAIN" if is_training else "TEST"
    print(f"[{phase}] Transmitting activations for {sensor_id}... Payload: {payload_size} bytes")

    # Send to server, receive gradient feedback
    response = stub.Forward(request)
    latency_ms = (time.time() - start_time) * 1000.0

    # Parse loss from server's status message, e.g. "Success: Loss 0.0040"
    # Parse loss from server's status message, e.g. "Success: Loss 0.0040 Pred 0.123"
    try:
        parts = response.status_message.split()
        loss_idx = parts.index("Loss") + 1
        current_loss = float(parts[loss_idx])
        
        if "Pred" in parts:
            pred_idx = parts.index("Pred") + 1
            prediction_val = float(parts[pred_idx])
        else:
            prediction_val = 0.0
    except Exception:
        current_loss = float('inf') # Default to infinity to avoid false 'best' model
        prediction_val = 0.0

    # Display progress
    icon = "💧💧💧" if target_value > 0 else "☁️"
    print(f"{icon} [{mode}] {sensor_id[:10]} | 3h Target: {target_value:.2f} | Loss: {current_loss:.6f}")

    # Decompress gradient and run backward pass ONLY if training
    if is_training:
        received_grad = decompress(response.gradient_data, smashed_activation.shape, compression_mode).to(device)
        smashed_activation.backward(received_grad)
        
        # Gradient Clipping to prevent explosions
        torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
        
        optimizer.step()

    print(f"[SERVER] Feedback processed | {response.status_message} | Latency: {latency_ms:.2f} ms")

    return {
        "Target": target_value,
        "Prediction": prediction_val,
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

def _save_results(client_id: int, experimental_logs: list, session_id: str, best_model_path: str = None, 
                  best_test_loss: float = None, avg_latency: float = None, avg_bytes: float = None) -> None:
    """
    Saves training logs as a CSV and a JSON metadata sidecar to results/.
    """
    output_dir = os.path.join(project_root, "results", session_id)
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
        "best_model_path": best_model_path,
        "best_test_loss": best_test_loss,
        "avg_latency_ms": avg_latency,
        "avg_payload_bytes": avg_bytes,
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

        # Step 1: Register with server to get an assigned ID + shared session_id
        reg = stub.Register(fsl_pb2.RegisterRequest())
        client_id, num_clients, session_id = reg.client_id, reg.total_clients, reg.session_id
        print(f"[CLIENT] Registered — ID: {client_id} / {num_clients} | session: {session_id}")

        # Shared session directory (same as server's)
        session_dir = os.path.join(project_root, "bestweights", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Step 2: Partition sensor files for this client
        all_files = sorted(glob.glob(os.path.join(project_root, data_dir, "*.parquet")))
        chunk_size = len(all_files) // num_clients
        start_idx  = (client_id - 1) * chunk_size
        end_idx    = start_idx + chunk_size if client_id < num_clients else len(all_files)
        client_files = all_files[start_idx:end_idx]
        print(f"[CLIENT {client_id}] Allocated {len(client_files)}/{len(all_files)} sensors")

        # Step 3: Initialize model and training state
        use_gpu = cfg.get("training", {}).get("use_gpu", False)
        device = torch.device("mps") if (use_gpu and torch.backends.mps.is_available()) else torch.device("cpu")
        print(f"[CLIENT {client_id}] Using device: {device} (use_gpu={use_gpu})")
        
        client_model = ClientLSTM(
            input_size=cfg.get("model", {}).get("input_size", len(FEATURE_COLS)),
            hidden_size=cfg.get("model", {}).get("hidden_size", 64),
        ).to(device)
        
        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=cfg.get("training", {}).get("lr", 0.001)
        )
        client_model.train()
        experimental_logs = []

        # Load all assigned sensor files into memory once to avoid disk I/O per epoch
        print(f"[CLIENT {client_id}] Pre-loading sensor data into memory...")
        sensor_data_cache = {}
        for file_path in client_files:
            sensor_id = Path(file_path).stem
            try:
                sensor_data_cache[file_path] = _load_sensor_data(file_path)
            except Exception as e:
                print(f"[CLIENT {client_id} ERROR] Failed to load {sensor_id}: {e}")
                
        # Track best performance for early stopping & checkpointing
        best_test_loss = float('inf')
        no_improvement_count = 0
        patience = cfg.get("training", {}).get("early_stopping_patience", 15)
        
        best_model_path = None
        current_round = 0  # track which FedAvg round we are on
        ckpt_interval = cfg.get("training", {}).get("checkpoint_interval", 10)
        periodic_dir  = os.path.join(session_dir, "periodic")
        os.makedirs(periodic_dir, exist_ok=True)
        os.makedirs(os.path.join(project_root, "results"), exist_ok=True)
        
        # Calculate global mean/std for all assigned sensors for normalization
        print(f"[CLIENT {client_id}] Calculating feature statistics for normalization...")
        all_combined = pd.concat(sensor_data_cache.values())
        feat_mean = all_combined[FEATURE_COLS].mean().values
        feat_std  = all_combined[FEATURE_COLS].std().values + 1e-9 # avoid div zero
        feat_stats = (feat_mean, feat_std)

        # Step 4: Training loop
        for epoch in range(epochs):
            print(f"[EPOCH {epoch+1}/{epochs}] Client {client_id} starting...")

            for file_path in client_files:
                optimizer.zero_grad()
                sensor_id = Path(file_path).stem

                try:
                    df = sensor_data_cache.get(file_path)
                    if df is None:
                        continue
                    result = _sample_index(df, is_training=True)

                    if result is None:
                        continue

                    target_idx, mode = result
                    target_value = float(df['future_3h_rain'].iloc[target_idx])

                    log_entry = _forward_step(
                        stub, client_id, client_model, optimizer,
                        df, target_idx, target_value, mode, sensor_id, 
                        compression_mode, feat_stats, is_training=True
                    )
                    experimental_logs.append({"Epoch": epoch + 1, "Status": "TRAIN", "Sensor": sensor_id, **log_entry})

                except Exception as e:
                    print(f"[CLIENT {client_id} ERROR] {sensor_id}: {e}")

            # End-of-epoch: sync weights with server
            print(f"[CLIENT {client_id}] Epoch {epoch+1} done. Synchronizing...")
            try:
                client_model = _fed_avg_sync(stub, client_id, client_model)
                current_round += 1  # increment round counter after successful sync
            except Exception as e:
                print(f"[CLIENT {client_id}] Sync failed: {e}")
                
            # Evaluation Phase (Testing on new global model)
            print(f"--- [EVALUATION] Client {client_id} Epoch {epoch+1} ---")
            client_model.eval()
            epoch_test_losses = []
            
            with torch.no_grad():
                for file_path in client_files:
                    sensor_id = Path(file_path).stem
                    try:
                        df = sensor_data_cache.get(file_path)
                        if df is None:
                            continue
                        result = _sample_index(df, is_training=False)
                        if result:
                            target_idx, mode = result
                            target_value = float(df['future_3h_rain'].iloc[target_idx])
                            log_entry = _forward_step(
                                stub, client_id, client_model, optimizer,
                                df, target_idx, target_value, mode, sensor_id, 
                                compression_mode, feat_stats, is_training=False
                            )
                            experimental_logs.append({"Epoch": epoch + 1, "Status": "TEST", "Sensor": sensor_id, **log_entry})
                            
                            if log_entry["Loss"] is not None:
                                epoch_test_losses.append(float(log_entry["Loss"]))
                    except Exception as e:
                        pass
                        
            # ── Checkpoint logic ─────────────────────────────────────────────
            if epoch_test_losses:
                avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                print(f"[CLIENT {client_id}] Epoch {epoch+1} Avg Test Loss: {avg_test_loss:.4f}")

                # Shared checkpoint metadata
                num_layers_ckpt = sum(
                    1 for k in client_model.state_dict()
                    if k.startswith("lstm.weight_ih_l")
                )
                base_ckpt = {
                    "round":                current_round,
                    "epoch":               epoch + 1,
                    "model_state_dict":    client_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                avg_test_loss,
                    "config": {
                        "hidden_size": cfg.get("model", {}).get("hidden_size", 64),
                        "num_layers":  num_layers_ckpt,
                        "input_size":  cfg.get("model", {}).get("input_size", 5),
                    },
                    "session_id": session_id,
                    "client_id":  client_id,
                }

                # ── 建議 1: Best checkpoint ──────────────────────────────────
                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    no_improvement_count = 0 # reset
                    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    best_model_path = os.path.join(
                        session_dir,
                        f"best_client_{client_id}_round_{current_round}_model_{stamp}.pth"
                    )
                    torch.save(base_ckpt, best_model_path)
                    print(f"\u2728 [CLIENT {client_id}] New Best! Round {current_round}, "
                          f"Loss={best_test_loss:.4f} \u2192 {session_id}/{Path(best_model_path).name}")
                else:
                    no_improvement_count += 1
                    print(f"[CLIENT {client_id}] No improvement for {no_improvement_count}/{patience} rounds.")

                # ── Early Stopping trigger ──────────────────────────────────
                if no_improvement_count >= patience:
                    print(f"\n🛑 [EARLY STOP] Client {client_id} triggered at round {current_round} (Patience={patience})")
                    break

                # ── Periodic checkpoint (every N rounds) ─────────────
                if current_round > 0 and current_round % ckpt_interval == 0:
                    periodic_path = os.path.join(
                        periodic_dir,
                        f"client_{client_id}_round_{current_round:04d}.pth"
                    )
                    torch.save(base_ckpt, periodic_path)
                    print(f"[CLIENT {client_id}] 💾 Periodic ckpt saved: round {current_round:04d}")
            
            # Check if outer loop should break (for Early Stopping)
            if no_improvement_count >= patience:
                break
            
    # Calculate some summary stats from the experimental logs
    num_logs = len(experimental_logs)
    total_latency = sum(float(log["LatencyMs"]) for log in experimental_logs)
    avg_latency = total_latency / num_logs if num_logs > 0 else 0.0
    total_bytes = sum(float(log["PayloadBytes"]) for log in experimental_logs)
    avg_bytes = total_bytes / num_logs if num_logs > 0 else 0.0

    # Step 5: Save results
    _save_results(
        client_id, 
        experimental_logs, 
        session_id,
        best_model_path, 
        best_test_loss=best_test_loss if best_test_loss != float('inf') else None,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes
    )
    
    print("\n" + "="*60)
    print(f"🏆  TRAINING COMPLETE: CLIENT {client_id} SUMMARY")
    print("="*60)
    print(f"[INFO]  Total Epochs Completed : {epochs}")
    print(f"[INFO]  Total Forward Passes   : {num_logs}")
    print(f"[INFO]  Best Test Loss (MSE)   : {best_test_loss:.4f}" if best_test_loss != float('inf') else "[INFO]  Best Test Loss (MSE)   : N/A")
    print(f"[INFO]  Avg Latency per Pass   : {avg_latency:.2f} ms")
    print(f"[INFO]  Avg Payload per Pass   : {avg_bytes/1024:.2f} KB")
    print(f"[INFO]  Best Model Checkpoint  : {best_model_path}")
    print("="*60 + "\n")

if __name__ == '__main__':
    run_all_client()
