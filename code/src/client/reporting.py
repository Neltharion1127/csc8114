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


def print_summary(
    *,
    client_id: int,
    epochs: int,
    num_logs: int,
    best_test_loss: float,
    avg_latency: float,
    avg_bytes: float,
    best_model_path: str | None,
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
    print(f"[INFO]  Best Model Checkpoint  : {best_model_path}")
    print("=" * 60 + "\n")
