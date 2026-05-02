"""
Generate results/matrix_summary.csv from per-scenario eval reports.

One row per scenario, metrics averaged across 11 client/sensor rows.
Uses:
  - Old session (2026-04-30_01-17-30): N01-N04, L05-L09, H11-H14  (_fixed38 variant)
  - New session (2026-04-30_16-04-59): L10, H15, H16, M17          (default, already thr=0.38)

Output columns: scenario_id, auprc_mean, auprc_std, roc_auc_mean, f1_mean,
                mse_mean, mae_mean, avg_payload_bytes, latency_ms_mean
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"

OLD = RESULTS / "2026-04-30_01-17-30"
NEW = RESULTS / "2026-04-30_16-04-59"

SOURCES = {
    "N01": (OLD, "N01_eval_report_fixed38.csv"),
    "N02": (OLD, "N02_eval_report_fixed38.csv"),
    "N03": (OLD, "N03_eval_report_fixed38.csv"),
    "N04": (OLD, "N04_eval_report_fixed38.csv"),
    "L05": (OLD, "L05_eval_report_fixed38.csv"),
    "L06": (OLD, "L06_eval_report_fixed38.csv"),
    "L07": (OLD, "L07_eval_report_fixed38.csv"),
    "L08": (OLD, "L08_eval_report_fixed38.csv"),
    "L09": (OLD, "L09_eval_report_fixed38.csv"),
    "L10": (NEW, "L10_eval_report_fixed38.csv"),
    "H11": (OLD, "H11_eval_report_fixed38.csv"),
    "H12": (OLD, "H12_eval_report_fixed38.csv"),
    "H13": (OLD, "H13_eval_report_fixed38.csv"),
    "H14": (OLD, "H14_eval_report_fixed38.csv"),
    "H15": (NEW, "H15_eval_report_fixed38.csv"),
    "H16": (NEW, "H16_eval_report_fixed38.csv"),
    "M17": (NEW, "M17_eval_report_fixed38.csv"),
}

METRICS = ["auprc", "roc_auc", "f1", "mse", "mae", "rain_mse", "rain_mae", "payload_bytes", "latency_ms"]

rows = []
for sid, (session_dir, fname) in SOURCES.items():
    path = session_dir / fname
    if not path.exists():
        print(f"[SKIP] {sid}: {path} not found")
        continue
    df = pd.read_csv(path)
    df = df[df["client_id"] != "SUMMARY"].copy()
    for col in METRICS:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    row = {"scenario_id": sid}
    row["auprc_mean"]        = df["auprc"].mean()
    row["auprc_std"]         = df["auprc"].std()
    row["roc_auc_mean"]      = df["roc_auc"].mean()
    row["f1_mean"]           = df["f1"].mean()
    row["mse_mean"]          = df["mse"].mean()
    row["mae_mean"]          = df["mae"].mean()
    row["rain_mse_mean"]     = df["rain_mse"].mean()
    row["rain_mae_mean"]     = df["rain_mae"].mean()
    row["avg_payload_bytes"] = df["payload_bytes"].mean()
    row["latency_ms_mean"]   = df["latency_ms"].mean()
    rows.append(row)
    print(f"[OK] {sid}: AUPRC={row['auprc_mean']:.4f}±{row['auprc_std']:.4f}  "
          f"payload={row['avg_payload_bytes']:.1f}B  F1={row['f1_mean']:.4f}")

out = RESULTS / "matrix_summary.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nSaved {len(rows)} scenarios → {out}")
