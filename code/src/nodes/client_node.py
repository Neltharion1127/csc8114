import grpc
import glob
import os
import re
import socket
import time
from pathlib import Path

import pandas as pd
import torch

from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.client.checkpointing import CheckpointState, evaluate_epoch
from src.client.data_pipeline import load_sensor_data, sample_index
from src.client.forward_step import run_forward_step
from src.client.reporting import print_summary, save_results, summarize_logs
from src.client.scheduler_state import SchedulerState
from src.client.sync import fed_avg_sync
from src.models.split_lstm import ClientLSTM

from src.shared.common import cfg, project_root

FEATURE_COLS = cfg.get("data", {}).get(
    "feature_cols",
    ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"],
)

end_date_str = cfg.get("data_download", {}).get("end_date", "2026-03-10T00:00:00")
test_days = cfg.get("data", {}).get("test_days", 14)
SPLIT_DATE = pd.Timestamp(end_date_str).tz_localize(None) - pd.Timedelta(days=test_days)


def _resolve_requested_client_id() -> int:
    """Prefer an explicit CLIENT_ID, otherwise derive it from a compose-style hostname."""
    raw_env = os.getenv("CLIENT_ID", "").strip()
    if raw_env:
        try:
            return int(raw_env)
        except ValueError:
            pass

    hostname = os.getenv("HOSTNAME") or socket.gethostname()
    match = re.fullmatch(r"fsl-client-(\d+)", hostname)
    return int(match.group(1)) if match else 0

def run_all_client(data_dir: str = "dataset/processed", epochs: int = 10) -> None:
    """Run the full client training and evaluation loop."""
    server_host = cfg.get("grpc", {}).get("server_host", "fsl-server")
    server_port = cfg.get("grpc", {}).get("server_port", 50051)
    target_address = f"{server_host}:{server_port}"
    compression_mode = cfg.get("compression", {}).get("mode", "float32")
    epochs = cfg.get("training", {}).get("num_rounds", epochs)
    client_name = os.getenv("HOSTNAME") or socket.gethostname()
    requested_client_id = _resolve_requested_client_id()

    time.sleep(cfg.get("training", {}).get("start_delay", 8))
    print(f"[CLIENT] Connecting to {target_address} for registration...")

    with grpc.insecure_channel(target_address) as channel:
        stub = fsl_pb2_grpc.FSLServiceStub(channel)

        # Step 1: Register with server to get an assigned ID + shared session_id
        reg = stub.Register(
            fsl_pb2.RegisterRequest(
                client_name=client_name,
                requested_client_id=requested_client_id,
            )
        )
        client_id, num_clients, session_id = reg.client_id, reg.total_clients, reg.session_id
        print(
            f"[CLIENT] Registered — name: {client_name} | requested_id: {requested_client_id or 'auto'} "
            f"| assigned_id: {client_id} / {num_clients} | session: {session_id}"
        )

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
        if not client_files:
            raise RuntimeError(
                f"Client {client_id} was assigned 0 sensors "
                f"(total_sensors={len(all_files)}, total_clients={num_clients}). "
                "Reduce num_clients or provide more sensor files."
            )

        use_gpu = cfg.get("training", {}).get("use_gpu", False)
        device = torch.device("mps") if (use_gpu and torch.backends.mps.is_available()) else torch.device("cpu")
        print(f"[CLIENT {client_id}] Using device: {device} (use_gpu={use_gpu})")
        
        client_model = ClientLSTM(
            input_size=cfg.get("model", {}).get("input_size", len(FEATURE_COLS)),
            hidden_size=cfg.get("model", {}).get("hidden_size", 64),
            num_layers=cfg.get("model", {}).get("num_layers", 1),
        ).to(device)
        
        optimizer = torch.optim.Adam(
            client_model.parameters(),
            lr=cfg.get("training", {}).get("lr", 0.001)
        )
        client_model.train()
        experimental_logs = []

        print(f"[CLIENT {client_id}] Pre-loading sensor data into memory...")
        sensor_data_cache = {}
        for file_path in client_files:
            sensor_id = Path(file_path).stem
            try:
                sensor_data_cache[file_path] = load_sensor_data(file_path)
            except Exception as e:
                print(f"[CLIENT {client_id} ERROR] Failed to load {sensor_id}: {e}")
        if not sensor_data_cache:
            raise RuntimeError(
                f"Client {client_id} could not load any sensor data from {len(client_files)} assigned files. "
                "Check dataset contents and preprocessing outputs."
            )

        checkpoint_state = CheckpointState()
        patience = cfg.get("training", {}).get("early_stopping_patience", 15)
        current_round = 0
        ckpt_interval = cfg.get("training", {}).get("checkpoint_interval", 10)
        periodic_dir = os.path.join(session_dir, "periodic")
        os.makedirs(periodic_dir, exist_ok=True)
        os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

        print(f"[CLIENT {client_id}] Calculating feature statistics for normalization...")
        all_combined = pd.concat(sensor_data_cache.values())
        feat_mean = all_combined[FEATURE_COLS].mean().values
        feat_std = all_combined[FEATURE_COLS].std().values + 1e-9
        feat_stats = (feat_mean, feat_std)

        train_state = SchedulerState(compression_mode=compression_mode)
        test_state = SchedulerState(compression_mode=compression_mode)

        for epoch in range(epochs):
            client_model.train()
            print(f"[EPOCH {epoch+1}/{epochs}] Client {client_id} starting...")
            epoch_train_losses = []
            epoch_train_steps = 0

            for file_path in client_files:
                optimizer.zero_grad()
                sensor_id = Path(file_path).stem

                try:
                    df = sensor_data_cache.get(file_path)
                    if df is None:
                        continue
                    result = sample_index(df, SPLIT_DATE, is_training=True)

                    if result is None:
                        continue

                    target_idx, mode = result
                    target_value = float(df['future_3h_rain'].iloc[target_idx])

                    log_entry = run_forward_step(
                        stub,
                        client_id,
                        client_model,
                        optimizer,
                        df,
                        target_idx,
                        target_value,
                        mode,
                        sensor_id,
                        train_state.compression_mode,
                        FEATURE_COLS,
                        feat_stats,
                        is_training=True,
                        last_latency_ms=train_state.last_latency_ms,
                    )
                    train_state.update(log_entry)
                    epoch_train_steps += 1
                    if log_entry["Loss"] is not None:
                        epoch_train_losses.append(float(log_entry["Loss"]))

                    experimental_logs.append({"Epoch": epoch + 1, "Status": "TRAIN", "Sensor": sensor_id, **log_entry})

                except Exception as e:
                    print(f"[CLIENT {client_id} ERROR] {sensor_id}: {e}")

            if epoch_train_steps:
                avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else float("nan")
                print(
                    f"[CLIENT {client_id}] Epoch {epoch+1} train summary | "
                    f"steps={epoch_train_steps} avg_loss={avg_train_loss:.4f}"
                )

            print(f"[CLIENT {client_id}] Epoch {epoch+1} done. Synchronizing...")
            try:
                client_model = fed_avg_sync(stub, client_id, client_model)
                current_round += 1
            except Exception as e:
                print(f"[CLIENT {client_id}] Sync failed: {e}")

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
                        result = sample_index(df, SPLIT_DATE, is_training=False)
                        if result:
                            target_idx, mode = result
                            target_value = float(df['future_3h_rain'].iloc[target_idx])
                            log_entry = run_forward_step(
                                stub,
                                client_id,
                                client_model,
                                optimizer,
                                df,
                                target_idx,
                                target_value,
                                mode,
                                sensor_id,
                                test_state.compression_mode,
                                FEATURE_COLS,
                                feat_stats,
                                is_training=False,
                                last_latency_ms=test_state.last_latency_ms,
                            )
                            test_state.update(log_entry)
                            experimental_logs.append({"Epoch": epoch + 1, "Status": "TEST", "Sensor": sensor_id, **log_entry})

                            if log_entry["Loss"] is not None:
                                epoch_test_losses.append(float(log_entry["Loss"]))
                    except Exception:
                        pass

            if epoch_test_losses:
                avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                print(
                    f"[CLIENT {client_id}] Epoch {epoch+1} test summary | "
                    f"steps={len(epoch_test_losses)} avg_loss={avg_test_loss:.4f}"
                )
                should_stop = evaluate_epoch(
                    client_id=client_id,
                    client_model=client_model,
                    optimizer=optimizer,
                    current_round=current_round,
                    epoch=epoch,
                    avg_test_loss=avg_test_loss,
                    session_id=session_id,
                    session_dir=session_dir,
                    periodic_dir=periodic_dir,
                    patience=patience,
                    ckpt_interval=ckpt_interval,
                    state=checkpoint_state,
                )
                if should_stop:
                    break

            if checkpoint_state.no_improvement_count >= patience:
                break

    num_logs = len(experimental_logs)
    avg_latency, avg_bytes = summarize_logs(experimental_logs)

    save_results(
        client_id,
        experimental_logs,
        session_id,
        best_model_path=checkpoint_state.best_model_path,
        best_test_loss=checkpoint_state.best_test_loss if checkpoint_state.best_test_loss != float("inf") else None,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes,
    )

    print_summary(
        client_id=client_id,
        epochs=epochs,
        num_logs=num_logs,
        best_test_loss=checkpoint_state.best_test_loss,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes,
        best_model_path=checkpoint_state.best_model_path,
    )

if __name__ == "__main__":
    run_all_client()
