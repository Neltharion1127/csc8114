"""
Generate results/matrix_summary.csv from per-scenario eval reports.

One row per scenario, metrics averaged across 11 client/sensor rows.
Uses session 2026-05-03_00-20-00 (all 17 scenarios, thr=0.5).

Output columns: scenario_id, auprc_mean, auprc_std, roc_auc_mean, f1_mean,
                mse_mean, mae_mean, avg_payload_bytes, latency_ms_mean
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "results"

SESSION = RESULTS / "2026-05-03_00-20-00"

SOURCES = {
    "N01": (SESSION, "N01_eval_report.csv"),
    "N02": (SESSION, "N02_eval_report.csv"),
    "N03": (SESSION, "N03_eval_report.csv"),
    "N04": (SESSION, "N04_eval_report.csv"),
    "L05": (SESSION, "L05_eval_report.csv"),
    "L06": (SESSION, "L06_eval_report.csv"),
    "L07": (SESSION, "L07_eval_report.csv"),
    "L08": (SESSION, "L08_eval_report.csv"),
    "L09": (SESSION, "L09_eval_report.csv"),
    "L10": (SESSION, "L10_eval_report.csv"),
    "H11": (SESSION, "H11_eval_report.csv"),
    "H12": (SESSION, "H12_eval_report.csv"),
    "H13": (SESSION, "H13_eval_report.csv"),
    "H14": (SESSION, "H14_eval_report.csv"),
    "H15": (SESSION, "H15_eval_report.csv"),
    "H16": (SESSION, "H16_eval_report.csv"),
    "M17": (SESSION, "M17_eval_report.csv"),
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
