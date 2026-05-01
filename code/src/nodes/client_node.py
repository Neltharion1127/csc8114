import grpc
import glob
import os
import re
import socket
import time
import psutil
from dataclasses import dataclass, field

import torch

from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.client.checkpointing import CheckpointState, evaluate_epoch
from src.client.data_pipeline import partition_client_files
from src.client.reporting import print_summary, save_progress, save_results, summarize_logs, summarize_phase
from src.client.scheduler_state import SchedulerState
from src.client.sync import fed_avg_sync
from src.client.training_loop import (
    build_eval_index_cache,
    compute_feature_stats,
    preload_sensor_data,
    run_eval_epoch,
    run_train_epoch,
)
from src.models.split_lstm import ClientLSTM
from src.shared.common import cfg, feature_cols_from_cfg, project_root
from src.shared.runtime import create_grpc_channel, resolve_device, resolve_server_address, set_global_seed
from src.shared.targets import rain_threshold_mm

FEATURE_COLS = feature_cols_from_cfg()

# Maximum reconnect attempts on transient gRPC failures.
_MAX_RECONNECT = 5
_RECONNECT_BACKOFF = [5, 15, 30, 60, 120]  # seconds between attempts


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_requested_client_id() -> int:
    """Prefer an explicit CLIENT_ID env var; fall back to compose-style hostname."""
    raw_env = os.getenv("CLIENT_ID", "").strip()
    if raw_env:
        try:
            return int(raw_env)
        except ValueError:
            pass
    hostname = os.getenv("HOSTNAME") or socket.gethostname()
    match = re.fullmatch(r"fsl-client-(\d+)", hostname)
    return int(match.group(1)) if match else 0


def _is_retriable(exc: grpc.RpcError) -> bool:
    """True for transient network errors that are safe to retry."""
    return exc.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN)


# ── Persistent state ──────────────────────────────────────────────────────────

    def get_system_metrics(self):
        """Monitor CPU and Memory utilization."""
        return {
            "CPU_Percent": psutil.cpu_percent(interval=None),
            "Mem_Percent": psutil.virtual_memory().percent
        }

    def get_model_size_bytes(self):
        """Calculate total model size in bytes."""
        param_size = 0
        for param in self.client_model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.client_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size

@dataclass
class _ClientState:
    """
    All mutable state for one client process.
    Fields set after registration and initialisation survive gRPC reconnects
    so training can resume from the last committed epoch.
    """
    # Set after registration
    client_id: int | None = None
    session_id: str | None = None
    num_clients: int = 0
    session_dir: str | None = None
    periodic_dir: str | None = None
    actual_seed: int | None = None

    # Set by _init_local (once per run, not per connection)
    client_model: ClientLSTM | None = None
    optimizer: torch.optim.Optimizer | None = None
    device: torch.device | None = None
    client_files: list | None = None
    sensor_data_cache: object = None
    feat_stats: object = None
    val_index_cache: object = None
    seq_len: int = 0
    target_horizon: int = 0
    train_state: SchedulerState | None = None
    val_state: SchedulerState | None = None

    # Progress (must survive reconnects)
    start_epoch: int = 0
    current_round: int = 0
    model_round: int = 0
    completed_epochs: int = 0
    total_steps: int = 0
    checkpoint_state: CheckpointState = field(default_factory=CheckpointState)
    experimental_logs: list = field(default_factory=list)
    finalized: bool = False
    run_start_time: float = field(default_factory=time.time)
    # Sync transmission tracking
    total_sync_bytes_sent: int = 0
    total_sync_bytes_recv: int = 0

    def get_system_metrics(self):
        """Monitor CPU, Memory (RSS), and Network utilization."""
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        net_info = psutil.net_io_counters()
        
        current_rss_mb = mem_info.rss / (1024 * 1024)
        if not hasattr(self, "_peak_rss_mb"):
            self._peak_rss_mb = current_rss_mb
        else:
            self._peak_rss_mb = max(self._peak_rss_mb, current_rss_mb)
            
        return {
            "CPU_Percent": psutil.cpu_percent(interval=None),
            "Mem_Percent": psutil.virtual_memory().percent,
            "Mem_RSS_MB": round(current_rss_mb, 2),
            "Mem_Peak_MB": round(self._peak_rss_mb, 2),
            "Net_Sent_MB": round(net_info.bytes_sent / (1024 * 1024), 2),
            "Net_Recv_MB": round(net_info.bytes_recv / (1024 * 1024), 2)
        }

    def get_model_size_bytes(self):
        """Calculate total model size in bytes."""
        if self.client_model is None:
            return 0
        param_size = sum(p.nelement() * p.element_size() for p in self.client_model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.client_model.buffers())
        return param_size + buffer_size


# ── Step 1: Register ──────────────────────────────────────────────────────────

def _register(stub, state: _ClientState, client_name: str, requested_client_id: int) -> None:
    """
    Registers the client with the server to get a logical ID and session.
    Retries indefinitely if a scenario mismatch is detected (waiting for server).
    """
    scenario_id = os.environ.get("SCENARIO_ID", "")
    while True:
        try:
            reg = stub.Register(
                fsl_pb2.RegisterRequest(
                    client_name=client_name,
                    requested_client_id=requested_client_id,
                ),
                metadata=[("scenario-id", scenario_id)]
            )
            
            if reg.session_id == "ERROR_SCENARIO_MISMATCH":
                print(f"[CLIENT] Waiting for server to switch to scenario: {scenario_id}...")
                time.sleep(10)
                continue
                
            new_client_id, num_clients, new_session_id = reg.client_id, reg.total_clients, reg.session_id
            break
            
        except grpc.RpcError as e:
            print(f"[CLIENT] Registration failed (server might be restarting): {e.code()}")
            time.sleep(10)

    if state.client_id is None:
        state.client_id = new_client_id
        state.session_id = new_session_id
        state.num_clients = num_clients
        base_seed = cfg.get("training", {}).get("seed", 42)
        client_seed = (int(base_seed) + int(state.client_id)) if base_seed is not None else None
        state.actual_seed = client_seed
        set_global_seed(client_seed, role=f"client-{state.client_id}")
        scenario_id = os.environ.get("SCENARIO_ID")
        if scenario_id:
            state.session_dir = os.path.join(project_root, "bestweights", state.session_id, scenario_id)
        else:
            state.session_dir = os.path.join(project_root, "bestweights", state.session_id)
        state.periodic_dir = os.path.join(state.session_dir, "periodic")
        os.makedirs(state.session_dir, exist_ok=True)
        os.makedirs(state.periodic_dir, exist_ok=True)
        # Pre-create results/<session_id>[/<scenario_id>] so save_results never
        # hits FileNotFoundError on the second (or later) seed run.
        if scenario_id:
            os.makedirs(os.path.join(project_root, "results", state.session_id, scenario_id), exist_ok=True)
        else:
            os.makedirs(os.path.join(project_root, "results", state.session_id), exist_ok=True)
    elif new_client_id != state.client_id or new_session_id != state.session_id:
        raise RuntimeError(
            f"Session mismatch on reconnect: "
            f"expected client={state.client_id}/session={state.session_id}, "
            f"got client={new_client_id}/session={new_session_id}. "
            "The server may have been restarted. Aborting."
        )

    print(
        f"[CLIENT] Registered name: {client_name} | requested_id: {requested_client_id or 'auto'} "
        f"| assigned_id: {state.client_id} / {num_clients} | session: {state.session_id}"
        f" | seed: {state.actual_seed}"
        + (" [resumed]" if state.start_epoch > 0 else "")
    )


# ── Step 2: One-time local setup ──────────────────────────────────────────────

def _init_local(state: _ClientState, data_dir: str, compression_mode: str) -> None:
    """
    Load sensor data, build model, compute feature statistics and val index cache.
    This is called once per training run; subsequent calls (on reconnect) are no-ops.
    """
    if state.client_model is not None:
        print(f"[CLIENT {state.client_id}] Resuming from epoch {state.start_epoch + 1}")
        return

    all_files = sorted(glob.glob(os.path.join(project_root, data_dir, "*.parquet")))
    state.client_files = partition_client_files(
        all_files, client_id=state.client_id, num_clients=state.num_clients,
    )
    print(f"[CLIENT {state.client_id}] Allocated {len(state.client_files)}/{len(all_files)} sensors")
    if not state.client_files:
        raise RuntimeError(
            f"Client {state.client_id} was assigned 0 sensors "
            f"(total_sensors={len(all_files)}, total_clients={state.num_clients}). "
            "Reduce num_clients or provide more sensor files."
        )

    state.device = resolve_device()
    print(f"[CLIENT {state.client_id}] Using device: {state.device}")

    model_cfg = cfg.get("model", {})
    lstm_dropout = float(model_cfg.get("lstm_dropout", model_cfg.get("dropout", 0.3)))
    state.client_model = ClientLSTM(
        input_size=model_cfg.get("input_size", len(FEATURE_COLS)),
        hidden_size=model_cfg.get("hidden_size", 64),
        num_layers=model_cfg.get("num_layers", 1),
        lstm_dropout=lstm_dropout,
    ).to(state.device)
    state.optimizer = torch.optim.Adam(
        state.client_model.parameters(),
        lr=cfg.get("training", {}).get("lr", 0.001),
    )

    state.target_horizon = max(1, int(cfg.get("model", {}).get("horizon", 3)))
    state.sensor_data_cache = preload_sensor_data(
        state.client_id, state.client_files, horizon=state.target_horizon,
    )
    state.feat_stats = compute_feature_stats(
        client_id=state.client_id,
        sensor_data_cache=state.sensor_data_cache,
        feature_cols=FEATURE_COLS,
    )
    state.seq_len = int(cfg.get("model", {}).get("seq_len", 24))
    eval_max_samples = max(0, int(cfg.get("training", {}).get("eval_max_samples_per_sensor", 0)))
    state.val_index_cache, val_samples, _ = build_eval_index_cache(
        client_id=state.client_id,
        sensor_data_cache=state.sensor_data_cache,
        target_phase="VAL",
        eval_max_samples=eval_max_samples,
        seq_len=state.seq_len,
        label="VAL",
        horizon=state.target_horizon,
    )
    if val_samples == 0:
        raise RuntimeError(
            f"Client {state.client_id} has 0 validation samples. "
            "Check dataset timestamps and train_end/val_end configuration."
        )
    data_cfg = cfg.get("data", {})
    str_train_end = data_cfg.get("train_end", "2024-12-31")
    str_val_end = data_cfg.get("val_end", "2025-06-30")
    print(
        f"[CLIENT {state.client_id}] Chronological split | TRAIN: <{str_train_end} "
        f"| VAL: {str_train_end}→<{str_val_end} | TEST: >={str_val_end} | horizon={state.target_horizon}h"
    )
    base_rho = max(1, int(cfg.get("federated", {}).get("rho", 1)))
    state.train_state = SchedulerState(compression_mode=compression_mode, rho=base_rho)
    state.val_state = SchedulerState(compression_mode=compression_mode, rho=base_rho)


# ── Step 3a: Validation (called after each sync) ──────────────────────────────

def _run_validation(
    stub,
    state: _ClientState,
    epoch: int,
    epoch_logs: list,
    patience: int,
    ckpt_interval: int,
) -> tuple[bool, int]:
    """
    Run one validation epoch and check early-stopping criteria.
    Returns (should_stop, val_step_count).
    """
    print(f"--- [VALIDATION] Client {state.client_id} Epoch {epoch+1} ---")
    state.client_model.eval()
    eval_start = time.time()
    epoch_val_losses, eval_metrics = run_eval_epoch(
        stub=stub,
        client_id=state.client_id,
        client_model=state.client_model,
        optimizer=state.optimizer,
        client_files=state.client_files,
        sensor_data_cache=state.sensor_data_cache,
        eval_index_cache=state.val_index_cache,
        eval_state=state.val_state,
        feature_cols=FEATURE_COLS,
        feat_stats=state.feat_stats,
        device=state.device,
        seq_len=state.seq_len,
        epoch=epoch,
        experimental_logs=state.experimental_logs,
        epoch_logs=epoch_logs,
        phase_label="VAL",
    )

    if not epoch_val_losses:
        return False, 0

    avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
    val_summary = summarize_phase(epoch_logs, "VAL")
    tp, fn, fp, tn = (
        int(eval_metrics["tp"]), int(eval_metrics["fn"]),
        int(eval_metrics["fp"]), int(eval_metrics["tn"]),
    )
    eval_elapsed = max(1e-9, time.time() - eval_start)
    print(
        f"[CLIENT {state.client_id}] Epoch {epoch+1} val summary | "
        f"steps={len(epoch_val_losses)} avg_loss={avg_val_loss:.4f} "
        f"rain_acc={val_summary['rain_acc']:.3f} "
        f"cls_loss={val_summary['avg_cls_loss']:.4f} reg_loss={val_summary['avg_reg_loss']:.4f} "
        f"positive_count={tp + fn} "
        f"recall={eval_metrics['recall']:.3f} precision={eval_metrics['precision']:.3f} "
        f"f1={eval_metrics['f1']:.3f} "
        f"thr={eval_metrics['selected_threshold']:.3f} "
        f"(default={eval_metrics['default_threshold']:.3f}, "
        f"default_r/p/f1={eval_metrics['default_recall']:.3f}/"
        f"{eval_metrics['default_precision']:.3f}/{eval_metrics['default_f1']:.3f}) "
        f"cm=TP:{tp}/FN:{fn}/FP:{fp}/TN:{tn} "
        f"val_time={eval_elapsed:.2f}s "
        f"val_throughput={len(epoch_val_losses) / eval_elapsed:.2f} steps/s"
    )

    should_stop = evaluate_epoch(
        client_id=state.client_id,
        client_model=state.client_model,
        optimizer=state.optimizer,
        current_round=state.current_round,
        epoch=epoch,
        avg_val_loss=avg_val_loss,
        val_metrics=eval_metrics,
        session_id=state.session_id,
        session_dir=state.session_dir,
        periodic_dir=state.periodic_dir,
        patience=patience,
        ckpt_interval=ckpt_interval,
        state=state.checkpoint_state,
    )
    return should_stop, len(epoch_val_losses)


# ── Step 3b: Single training epoch ───────────────────────────────────────────

def _run_single_epoch(stub, state: _ClientState, epoch: int, epochs: int) -> bool:
    """
    Run one full epoch: train → optional FedAvg sync → optional validation.
    Updates *state* in-place. Returns True if training should stop.
    """
    training_cfg = cfg.get("training", {})
    patience = training_cfg.get("early_stopping_patience", 15)
    ckpt_interval = training_cfg.get("checkpoint_interval", 10)
    local_steps = max(1, int(training_cfg.get("local_steps", 1)))
    rain_sample_ratio = float(training_cfg.get("rain_sample_ratio", 0.35))
    rain_threshold = rain_threshold_mm()

    epoch_start = time.time()
    state.client_model.train()
    print(f"[EPOCH {epoch+1}/{epochs}] Client {state.client_id} starting...")
    epoch_logs: list[dict] = []

    # --- Train ---
    epoch_train_steps = run_train_epoch(
        stub=stub,
        client_id=state.client_id,
        client_model=state.client_model,
        optimizer=state.optimizer,
        client_files=state.client_files,
        sensor_data_cache=state.sensor_data_cache,
        train_state=state.train_state,
        feature_cols=FEATURE_COLS,
        feat_stats=state.feat_stats,
        device=state.device,
        local_steps=local_steps,
        rain_sample_ratio=rain_sample_ratio,
        seq_len=state.seq_len,
        epoch=epoch,
        experimental_logs=state.experimental_logs,
        epoch_logs=epoch_logs,
        horizon=state.target_horizon,
        rain_threshold=rain_threshold,
    )
    if epoch_train_steps:
        summary = summarize_phase(epoch_logs, "TRAIN")
        train_elapsed = max(1e-9, time.time() - epoch_start)
        print(
            f"[CLIENT {state.client_id}] Epoch {epoch+1} train summary | "
            f"steps={epoch_train_steps} avg_loss={summary['avg_loss']:.4f} "
            f"rain_acc={summary['rain_acc']:.3f} "
            f"cls_loss={summary['avg_cls_loss']:.4f} reg_loss={summary['avg_reg_loss']:.4f} "
            f"train_time={train_elapsed:.2f}s "
            f"train_throughput={epoch_train_steps / train_elapsed:.2f} steps/s"
        )
        
        # Log system metrics and epoch timing
        metrics = state.get_system_metrics()
        metrics["Epoch_Time_s"] = train_elapsed
        metrics["Model_Size_Bytes"] = state.get_model_size_bytes()
        for log in epoch_logs:
            log.update(metrics)

    # --- Sync + validate (every rho epochs) ---
    sync_interval = max(1, int(state.train_state.rho))
    val_steps = 0

    if (epoch + 1) % sync_interval == 0:
        print(f"[CLIENT {state.client_id}] Epoch {epoch+1} done. Synchronizing (rho={sync_interval})...")
        try:
            sync_result = fed_avg_sync(
                stub, state.client_id, state.client_model,
                model_round=state.model_round, local_epochs=sync_interval,
            )
            state.client_model = sync_result.client_model
            state.current_round = max(state.current_round, sync_result.round_number)
            state.model_round = max(state.model_round, sync_result.round_number)
            state.total_sync_bytes_sent += sync_result.sync_bytes_sent
            state.total_sync_bytes_recv += sync_result.sync_bytes_recv
        except Exception as exc:
            print(f"[CLIENT {state.client_id}] Sync failed: {exc}")
            if "Timeout waiting for global model aggregation" in str(exc):
                state.completed_epochs = epoch + 1
                state.total_steps += epoch_train_steps
                print(f"[CLIENT {state.client_id}] Stopping due to sync timeout after epoch {epoch+1}.")
                return True
            raise  # re-raise gRPC errors so the reconnect loop can catch them

        should_stop, val_steps = _run_validation(
            stub, state, epoch, epoch_logs, patience, ckpt_interval,
        )
        if should_stop:
            return True
    else:
        print(
            f"[CLIENT {state.client_id}] Epoch {epoch+1} done. "
            f"Skip sync (rho={sync_interval}); continuing local training."
        )

    # --- Commit epoch progress ---
    epoch_elapsed = max(1e-9, time.time() - epoch_start)
    epoch_steps = epoch_train_steps + val_steps
    print(
        f"[CLIENT {state.client_id}] Epoch {epoch+1} timing | "
        f"total_time={epoch_elapsed:.2f}s total_steps={epoch_steps} "
        f"throughput={epoch_steps / epoch_elapsed:.2f} steps/s"
    )
    state.start_epoch = epoch + 1   # reconnect resumes here
    state.completed_epochs = epoch + 1
    state.total_steps += epoch_steps

    avg_latency, avg_bytes = summarize_logs(state.experimental_logs)
    save_progress(
        state.client_id, state.experimental_logs, state.session_id,
        epoch=epoch + 1,
        best_model_path=state.checkpoint_state.best_model_path,
        best_test_loss=state.checkpoint_state.best_test_loss if state.checkpoint_state.best_test_loss != float("inf") else None,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes,
        model_size_bytes=state.get_model_size_bytes(),
    )
    return False


# ── Step 4: Finalise session ──────────────────────────────────────────────────

def _finalize_session(stub, state: _ClientState, epochs: int) -> None:
    """Save final results, print summary, and notify the server of completion."""
    # Calculate avg system metrics
    cpus = [log.get("CPU_Percent", 0.0) for log in state.experimental_logs if "CPU_Percent" in log]
    mems = [log.get("Mem_Percent", 0.0) for log in state.experimental_logs if "Mem_Percent" in log]
    avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
    avg_mem = sum(mems) / len(mems) if mems else 0.0
    
    # Track Peak Memory and Total Network
    final_metrics = state.get_system_metrics()
    net_sent_mb = final_metrics["Net_Sent_MB"]
    net_recv_mb = final_metrics["Net_Recv_MB"]
    mem_peak_mb = final_metrics["Mem_Peak_MB"]

    total_runtime = time.time() - state.run_start_time
    avg_latency, avg_bytes = summarize_logs(state.experimental_logs)

    save_results(
        state.client_id, state.experimental_logs, state.session_id,
        best_model_path=state.checkpoint_state.best_model_path,
        best_test_loss=state.checkpoint_state.best_test_loss if state.checkpoint_state.best_test_loss != float("inf") else None,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes,
        avg_cpu=avg_cpu,
        avg_mem=avg_mem,
        total_runtime_s=total_runtime,
        model_size_bytes=state.get_model_size_bytes(),
        net_sent_mb=net_sent_mb,
        net_recv_mb=net_recv_mb,
        mem_peak_mb=mem_peak_mb,
        sync_bytes_sent_mb=round(state.total_sync_bytes_sent / (1024 * 1024), 4),
        sync_bytes_recv_mb=round(state.total_sync_bytes_recv / (1024 * 1024), 4),
        actual_seed=state.actual_seed,
    )
    print_summary(
        client_id=state.client_id,
        epochs=state.completed_epochs or epochs,
        num_logs=len(state.experimental_logs),
        best_test_loss=state.checkpoint_state.best_test_loss,
        avg_latency=avg_latency,
        avg_bytes=avg_bytes,
        best_model_path=state.checkpoint_state.best_model_path,
        total_runtime_s=total_runtime,
        avg_steps_per_s=state.total_steps / max(1e-9, total_runtime),
        avg_cpu=avg_cpu,
        avg_mem=avg_mem,
        actual_seed=state.actual_seed,
    )
    completion = stub.NotifyCompletion(
        fsl_pb2.CompletionRequest(
            client_id=state.client_id,
            completed_epochs=state.completed_epochs or epochs,
            total_steps=state.total_steps,
            session_id=state.session_id,
        ),
        metadata=[("scenario-id", os.environ.get("SCENARIO_ID", ""))]
    )
    print(
        f"[CLIENT {state.client_id}] Completion acknowledged by server "
        f"({completion.completed_clients}/{completion.total_clients})"
    )
    state.finalized = True


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all_client(data_dir: str = "dataset/processed", epochs: int = 10) -> None:
    """
    Orchestrate the full client lifecycle:
      1. Connect → register → init local data & model (once)
      2. Train for N epochs, syncing with server every rho rounds
      3. Finalise results and notify completion
    On transient gRPC failures the connection is re-established and training
    resumes from the last committed epoch (up to _MAX_RECONNECT attempts).
    """
    # Prevent PyTorch from spawning multiple BLAS threads per process.
    # With 11 client processes on 12 vCPUs, each process should own exactly
    # one core; letting PyTorch use all 12 threads causes 132-thread contention
    # and turns a 15 ms LSTM backward into 7+ seconds.
    torch.set_num_threads(max(1, int(cfg.get("training", {}).get("torch_num_threads", 1))))
    target_address = resolve_server_address()
    compression_mode = cfg.get("compression", {}).get("mode", "float32")
    epochs = cfg.get("training", {}).get("num_rounds", epochs)
    requested_client_id = _resolve_requested_client_id()
    # Fixed name so the server can match this client across reconnects.
    # (Docker container hostnames change on restart.)
    client_name = (
        f"fsl-client-cid{requested_client_id}"
        if requested_client_id > 0
        else (os.getenv("HOSTNAME") or socket.gethostname())
    )

    time.sleep(cfg.get("training", {}).get("start_delay", 8))

    state = _ClientState()

    for attempt in range(_MAX_RECONNECT):
        try:
            print(
                f"[CLIENT] Connecting to {target_address}"
                + (f" (attempt {attempt + 1}/{_MAX_RECONNECT})" if attempt > 0 else "") + "..."
            )
            with create_grpc_channel(target_address) as channel:
                stub = fsl_pb2_grpc.FSLServiceStub(channel)

                _register(stub, state, client_name, requested_client_id)
                _init_local(state, data_dir, compression_mode)

                for epoch in range(state.start_epoch, epochs):
                    if _run_single_epoch(stub, state, epoch, epochs):
                        break

                _finalize_session(stub, state, epochs)
                break  # success — exit reconnect loop

        except KeyboardInterrupt:
            print("[CLIENT] Interrupted by user; saving partial results...")
            break
        except grpc.RpcError as exc:
            if _is_retriable(exc) and attempt < _MAX_RECONNECT - 1:
                backoff = _RECONNECT_BACKOFF[attempt]
                print(
                    f"[CLIENT] Connection lost (attempt {attempt + 1}/{_MAX_RECONNECT}): "
                    f"{exc.details()}. Reconnecting in {backoff}s..."
                )
                time.sleep(backoff)
            else:
                print(f"[CLIENT] Fatal gRPC error after {attempt + 1} attempt(s): {exc.details()}")
                break
        except Exception as exc:
            print(f"[CLIENT] Fatal error: {exc}")
            break

    if not state.finalized and state.client_id is not None and state.session_id is not None:
        avg_latency, avg_bytes = summarize_logs(state.experimental_logs)
        cpus = [log.get("CPU_Percent", 0.0) for log in state.experimental_logs if "CPU_Percent" in log]
        mems = [log.get("Mem_Percent", 0.0) for log in state.experimental_logs if "Mem_Percent" in log]
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
        avg_mem = sum(mems) / len(mems) if mems else 0.0
        total_runtime = time.time() - state.run_start_time

        save_results(
            state.client_id, state.experimental_logs, state.session_id,
            best_model_path=state.checkpoint_state.best_model_path,
            best_test_loss=state.checkpoint_state.best_test_loss if state.checkpoint_state.best_test_loss != float("inf") else None,
            avg_latency=avg_latency,
            avg_bytes=avg_bytes,
            avg_cpu=avg_cpu,
            avg_mem=avg_mem,
            total_runtime_s=total_runtime,
            model_size_bytes=state.get_model_size_bytes(),
            actual_seed=state.actual_seed,
        )


if __name__ == "__main__":
    # Prioritise num_rounds from config, fallback to 10
    total_rounds = cfg.get("training", {}).get("num_rounds", 10)
    run_all_client(epochs=total_rounds)
