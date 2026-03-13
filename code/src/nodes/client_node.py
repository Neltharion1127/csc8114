import grpc
import glob
import os
import re
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.client.checkpointing import CheckpointState, evaluate_epoch
from src.client.data_pipeline import collect_test_indices, load_sensor_data, sample_index
from src.client.forward_step import run_forward_step
from src.client.reporting import print_summary, save_progress, save_results, summarize_logs, summarize_phase
from src.client.scheduler_state import SchedulerState
from src.client.sync import fed_avg_sync
from src.models.split_lstm import ClientLSTM

from src.shared.common import cfg, project_root
from src.shared.runtime import grpc_channel_options, resolve_device, resolve_server_address, set_global_seed
from src.shared.targets import is_rain

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
    target_address = resolve_server_address()
    compression_mode = cfg.get("compression", {}).get("mode", "float32")
    epochs = cfg.get("training", {}).get("num_rounds", epochs)
    requested_client_id = _resolve_requested_client_id()
    base_client_name = os.getenv("HOSTNAME") or socket.gethostname()
    client_name = (
        f"{base_client_name}-cid{requested_client_id}"
        if requested_client_id > 0
        else base_client_name
    )

    time.sleep(cfg.get("training", {}).get("start_delay", 8))
    print(f"[CLIENT] Connecting to {target_address} for registration...")

    channel_options = grpc_channel_options()
    client_id: int | None = None
    session_id: str | None = None
    checkpoint_state = CheckpointState()
    experimental_logs = []
    finalized = False
    run_start_time = time.time()
    completed_epochs = 0
    total_steps = 0

    try:
        with grpc.insecure_channel(target_address, options=channel_options) as channel:
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
            base_seed = cfg.get("training", {}).get("seed", 42)
            client_seed = (int(base_seed) + int(client_id)) if base_seed is not None else None
            set_global_seed(client_seed, role=f"client-{client_id}")

            # Shared session directory (same as server's)
            session_dir = os.path.join(project_root, "bestweights", session_id)
            os.makedirs(session_dir, exist_ok=True)

            # Step 2: Partition sensor files for this client
            all_files = sorted(glob.glob(os.path.join(project_root, data_dir, "*.parquet")))
            chunk_size = len(all_files) // num_clients
            start_idx = (client_id - 1) * chunk_size
            end_idx = start_idx + chunk_size if client_id < num_clients else len(all_files)
            client_files = all_files[start_idx:end_idx]
            print(f"[CLIENT {client_id}] Allocated {len(client_files)}/{len(all_files)} sensors")
            if not client_files:
                raise RuntimeError(
                    f"Client {client_id} was assigned 0 sensors "
                    f"(total_sensors={len(all_files)}, total_clients={num_clients}). "
                    "Reduce num_clients or provide more sensor files."
                )

            device = resolve_device()
            print(f"[CLIENT {client_id}] Using device: {device}")
            client_model = ClientLSTM(
                input_size=cfg.get("model", {}).get("input_size", len(FEATURE_COLS)),
                hidden_size=cfg.get("model", {}).get("hidden_size", 64),
                num_layers=cfg.get("model", {}).get("num_layers", 1),
            ).to(device)
            optimizer = torch.optim.Adam(
                client_model.parameters(),
                lr=cfg.get("training", {}).get("lr", 0.001),
            )
            client_model.train()

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

            patience = cfg.get("training", {}).get("early_stopping_patience", 15)
            current_round = 0
            ckpt_interval = cfg.get("training", {}).get("checkpoint_interval", 10)
            local_steps = max(1, int(cfg.get("training", {}).get("local_steps", 1)))
            rain_sample_ratio = float(cfg.get("training", {}).get("rain_sample_ratio", 0.35))
            periodic_dir = os.path.join(session_dir, "periodic")
            os.makedirs(periodic_dir, exist_ok=True)
            os.makedirs(os.path.join(project_root, "results"), exist_ok=True)

            print(f"[CLIENT {client_id}] Calculating feature statistics for normalization...")
            all_combined = pd.concat(sensor_data_cache.values())
            feat_mean = all_combined[FEATURE_COLS].mean().values
            feat_std = all_combined[FEATURE_COLS].std().values + 1e-9
            feat_stats = (feat_mean, feat_std)

            eval_max_samples = max(0, int(cfg.get("training", {}).get("eval_max_samples_per_sensor", 0)))
            test_index_cache: dict[str, np.ndarray] = {}
            total_eval_samples = 0
            total_eval_positive = 0
            for file_path, df in sensor_data_cache.items():
                test_indices = collect_test_indices(df, SPLIT_DATE)
                if eval_max_samples > 0 and len(test_indices) > eval_max_samples:
                    picks = np.linspace(0, len(test_indices) - 1, eval_max_samples, dtype=int)
                    test_indices = test_indices[picks]
                test_index_cache[file_path] = test_indices
                total_eval_samples += int(len(test_indices))
                if len(test_indices) > 0:
                    total_eval_positive += int(df["future_3h_rain"].iloc[test_indices].apply(is_rain).sum())
            print(
                f"[CLIENT {client_id}] Fixed test set prepared: "
                f"samples={total_eval_samples} positives={total_eval_positive} "
                f"(per_sensor_cap={eval_max_samples if eval_max_samples > 0 else 'FULL'})"
            )

            train_state = SchedulerState(compression_mode=compression_mode)
            test_state = SchedulerState(compression_mode=compression_mode)

            for epoch in range(epochs):
                epoch_start_time = time.time()
                client_model.train()
                print(f"[EPOCH {epoch+1}/{epochs}] Client {client_id} starting...")
                epoch_logs: list[dict] = []
                epoch_train_steps = 0
                sync_timeout_triggered = False

                for file_path in client_files:
                    sensor_id = Path(file_path).stem
                    try:
                        df = sensor_data_cache.get(file_path)
                        if df is None:
                            continue
                        for _ in range(local_steps):
                            optimizer.zero_grad()
                            result = sample_index(
                                df,
                                SPLIT_DATE,
                                is_training=True,
                                rain_sample_ratio=rain_sample_ratio,
                            )
                            if result is None:
                                continue
                            target_idx, mode = result
                            target_value = float(df["future_3h_rain"].iloc[target_idx])
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
                                device,
                                is_training=True,
                                last_latency_ms=train_state.last_latency_ms,
                            )
                            train_state.update(log_entry)
                            epoch_train_steps += 1
                            epoch_record = {"Epoch": epoch + 1, "Status": "TRAIN", "Sensor": sensor_id, **log_entry}
                            experimental_logs.append(epoch_record)
                            epoch_logs.append(epoch_record)
                    except Exception as e:
                        print(f"[CLIENT {client_id} ERROR] {sensor_id}: {e}")

                if epoch_train_steps:
                    train_summary = summarize_phase(epoch_logs, "TRAIN")
                    train_elapsed_s = max(1e-9, time.time() - epoch_start_time)
                    print(
                        f"[CLIENT {client_id}] Epoch {epoch+1} train summary | "
                        f"steps={epoch_train_steps} avg_loss={train_summary['avg_loss']:.4f} "
                        f"rain_acc={train_summary['rain_acc']:.3f} "
                        f"cls_loss={train_summary['avg_cls_loss']:.4f} "
                        f"reg_loss={train_summary['avg_reg_loss']:.4f} "
                        f"train_time={train_elapsed_s:.2f}s "
                        f"train_throughput={epoch_train_steps / train_elapsed_s:.2f} steps/s"
                    )

                print(f"[CLIENT {client_id}] Epoch {epoch+1} done. Synchronizing...")
                try:
                    client_model = fed_avg_sync(stub, client_id, client_model)
                    current_round += 1
                except Exception as e:
                    print(f"[CLIENT {client_id}] Sync failed: {e}")
                    if "Timeout waiting for global model aggregation" in str(e):
                        sync_timeout_triggered = True
                        completed_epochs = epoch + 1
                        total_steps += epoch_train_steps
                        print(
                            f"[CLIENT {client_id}] Stopping training due to synchronization timeout "
                            f"after epoch {epoch+1}. Other clients likely finished early."
                        )
                        break

                if sync_timeout_triggered:
                    break

                print(f"--- [EVALUATION] Client {client_id} Epoch {epoch+1} ---")
                client_model.eval()
                epoch_test_losses = []
                tp = fn = fp = tn = 0
                eval_start_time = time.time()
                with torch.no_grad():
                    for file_path in client_files:
                        sensor_id = Path(file_path).stem
                        try:
                            df = sensor_data_cache.get(file_path)
                            if df is None:
                                continue
                            test_indices = test_index_cache.get(file_path)
                            if test_indices is None or len(test_indices) == 0:
                                continue
                            for target_idx in test_indices:
                                target_value = float(df["future_3h_rain"].iloc[target_idx])
                                log_entry = run_forward_step(
                                    stub,
                                    client_id,
                                    client_model,
                                    optimizer,
                                    df,
                                    int(target_idx),
                                    target_value,
                                    "FIXED_TEST",
                                    sensor_id,
                                    test_state.compression_mode,
                                    FEATURE_COLS,
                                    feat_stats,
                                    device,
                                    is_training=False,
                                    last_latency_ms=test_state.last_latency_ms,
                                )
                                test_state.update(log_entry)
                                epoch_record = {"Epoch": epoch + 1, "Status": "TEST", "Sensor": sensor_id, **log_entry}
                                experimental_logs.append(epoch_record)
                                epoch_logs.append(epoch_record)
                                if log_entry["Loss"] is not None:
                                    epoch_test_losses.append(float(log_entry["Loss"]))
                                true_rain = is_rain(float(log_entry["Target"]))
                                pred_rain = is_rain(float(log_entry["Prediction"]))
                                if true_rain and pred_rain:
                                    tp += 1
                                elif true_rain and not pred_rain:
                                    fn += 1
                                elif not true_rain and pred_rain:
                                    fp += 1
                                else:
                                    tn += 1
                        except Exception:
                            pass

                if epoch_test_losses:
                    avg_test_loss = sum(epoch_test_losses) / len(epoch_test_losses)
                    test_summary = summarize_phase(epoch_logs, "TEST")
                    positive_count = tp + fn
                    recall = (tp / positive_count) if positive_count > 0 else None
                    eval_elapsed_s = max(1e-9, time.time() - eval_start_time)
                    print(
                        f"[CLIENT {client_id}] Epoch {epoch+1} test summary | "
                        f"steps={len(epoch_test_losses)} avg_loss={avg_test_loss:.4f} "
                        f"rain_acc={test_summary['rain_acc']:.3f} "
                        f"cls_loss={test_summary['avg_cls_loss']:.4f} "
                        f"reg_loss={test_summary['avg_reg_loss']:.4f} "
                        f"positive_count={positive_count} "
                        f"recall={(f'{recall:.3f}' if recall is not None else 'N/A')} "
                        f"cm=TP:{tp}/FN:{fn}/FP:{fp}/TN:{tn} "
                        f"test_time={eval_elapsed_s:.2f}s "
                        f"test_throughput={len(epoch_test_losses) / eval_elapsed_s:.2f} steps/s"
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

                epoch_elapsed_s = max(1e-9, time.time() - epoch_start_time)
                epoch_steps = epoch_train_steps + len(epoch_test_losses)
                print(
                    f"[CLIENT {client_id}] Epoch {epoch+1} timing | "
                    f"total_time={epoch_elapsed_s:.2f}s total_steps={epoch_steps} "
                    f"throughput={epoch_steps / epoch_elapsed_s:.2f} steps/s"
                )
                completed_epochs = epoch + 1
                total_steps += epoch_steps

                avg_latency, avg_bytes = summarize_logs(experimental_logs)
                save_progress(
                    client_id,
                    experimental_logs,
                    session_id,
                    epoch=epoch + 1,
                    best_model_path=checkpoint_state.best_model_path,
                    best_test_loss=checkpoint_state.best_test_loss if checkpoint_state.best_test_loss != float("inf") else None,
                    avg_latency=avg_latency,
                    avg_bytes=avg_bytes,
                )

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
                epochs=completed_epochs or epochs,
                num_logs=num_logs,
                best_test_loss=checkpoint_state.best_test_loss,
                avg_latency=avg_latency,
                avg_bytes=avg_bytes,
                best_model_path=checkpoint_state.best_model_path,
                total_runtime_s=time.time() - run_start_time,
                avg_steps_per_s=(total_steps / max(1e-9, time.time() - run_start_time)),
            )
            completion = stub.NotifyCompletion(
                fsl_pb2.CompletionRequest(
                    client_id=client_id,
                    completed_epochs=completed_epochs or epochs,
                    total_steps=total_steps,
                    session_id=session_id,
                )
            )
            print(
                f"[CLIENT {client_id}] Completion acknowledged by server "
                f"({completion.completed_clients}/{completion.total_clients})"
            )
            finalized = True

    except KeyboardInterrupt:
        print("[CLIENT] Interrupted by user; saving partial results...")
    except Exception as e:
        print(f"[CLIENT] Fatal error: {e}")
    finally:
        if not finalized and client_id is not None and session_id is not None:
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

if __name__ == "__main__":
    run_all_client()
