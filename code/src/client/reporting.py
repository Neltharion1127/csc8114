import json
import os
import time

import pandas as pd

from src.shared.common import cfg, project_root


def summarize_logs(experimental_logs: list[dict]) -> tuple[float, float]:
    num_logs = len(experimental_logs)
    total_latency = sum(float(log["LatencyMs"]) for log in experimental_logs)
    total_bytes = sum(float(log["PayloadBytes"]) for log in experimental_logs)
    avg_latency = total_latency / num_logs if num_logs > 0 else 0.0
    avg_bytes = total_bytes / num_logs if num_logs > 0 else 0.0
    return avg_latency, avg_bytes


def summarize_phase(logs: list[dict], phase: str) -> dict[str, float]:
    """Compute compact per-phase metrics for client console summaries."""
    phase_logs = [log for log in logs if log.get("Status") == phase]
    steps = len(phase_logs)
    if steps == 0:
        return {
            "steps": 0,
            "avg_loss": float("nan"),
            "rain_acc": float("nan"),
            "avg_cls_loss": float("nan"),
            "avg_reg_loss": float("nan"),
        }

    return {
        "steps": steps,
        "avg_loss": sum(float(log["Loss"]) for log in phase_logs) / steps,
        "rain_acc": sum(int((float(log["Target"]) > 0.1) == (float(log["Prediction"]) > 0.1)) for log in phase_logs) / steps,
        "avg_cls_loss": sum(float(log.get("ClassificationLoss", 0.0)) for log in phase_logs) / steps,
        "avg_reg_loss": sum(float(log.get("RegressionLoss", 0.0)) for log in phase_logs) / steps,
    }


def save_results(
    client_id: int,
    experimental_logs: list,
    session_id: str,
    *,
    best_model_path: str | None = None,
    best_test_loss: float | None = None,
    avg_latency: float | None = None,
    avg_bytes: float | None = None,
) -> None:
    output_dir = os.path.join(project_root, "results", session_id)
    os.makedirs(output_dir, exist_ok=True)

    log_df = pd.DataFrame(experimental_logs)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
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
        "profiler_enabled": cfg.get("profiler", {}).get("enabled", True),
        "scheduler_enabled": cfg.get("scheduler", {}).get("enabled", True),
        "cfg": cfg,
    }
    meta_path = filepath.replace(".csv", "_meta.json")
    try:
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
        print(f"[CLIENT] Saved metadata to {meta_path}")
    except Exception as e:
        print(f"[CLIENT WARN] Failed to write metadata: {e}")


def save_progress(
    client_id: int,
    experimental_logs: list,
    session_id: str,
    *,
    epoch: int,
    best_model_path: str | None = None,
    best_test_loss: float | None = None,
    avg_latency: float | None = None,
    avg_bytes: float | None = None,
) -> None:
    """
    Persist rolling progress to deterministic filenames so partial runs are recoverable.
    The file is overwritten each call and contains all logs collected so far.
    """
    output_dir = os.path.join(project_root, "results", session_id)
    os.makedirs(output_dir, exist_ok=True)

    csv_name = f"training_log_client{client_id}_progress.csv"
    csv_path = os.path.join(output_dir, csv_name)
    pd.DataFrame(experimental_logs).to_csv(csv_path, index=False)

    meta = {
        "epoch": epoch,
        "csv": csv_name,
        "num_records": len(experimental_logs),
        "best_model_path": best_model_path,
        "best_test_loss": best_test_loss,
        "avg_latency_ms": avg_latency,
        "avg_payload_bytes": avg_bytes,
        "profiler_enabled": cfg.get("profiler", {}).get("enabled", True),
        "scheduler_enabled": cfg.get("scheduler", {}).get("enabled", True),
        "is_partial": True,
    }
    meta_path = os.path.join(output_dir, f"training_log_client{client_id}_progress_meta.json")
    try:
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
    except Exception as e:
        print(f"[CLIENT WARN] Failed to write progress metadata: {e}")


def print_summary(
    *,
    client_id: int,
    epochs: int,
    num_logs: int,
    best_test_loss: float,
    avg_latency: float,
    avg_bytes: float,
    best_model_path: str | None,
    total_runtime_s: float | None = None,
    avg_steps_per_s: float | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE: CLIENT {client_id} SUMMARY")
    print("=" * 60)
    print(f"[INFO]  Total Epochs Completed : {epochs}")
    print(f"[INFO]  Total Forward Passes   : {num_logs}")
    print(
        f"[INFO]  Best Test Loss (MSE)   : {best_test_loss:.4f}"
        if best_test_loss != float("inf")
        else "[INFO]  Best Test Loss (MSE)   : N/A"
    )
    print(f"[INFO]  Avg Latency per Pass   : {avg_latency:.2f} ms")
    print(f"[INFO]  Avg Payload per Pass   : {avg_bytes / 1024:.2f} KB")
    if total_runtime_s is not None:
        print(f"[INFO]  Total Runtime         : {total_runtime_s:.2f} s")
    if avg_steps_per_s is not None:
        print(f"[INFO]  Avg Throughput        : {avg_steps_per_s:.2f} steps/s")
    print(f"[INFO]  Best Model Checkpoint  : {best_model_path}")
    print("=" * 60 + "\n")
