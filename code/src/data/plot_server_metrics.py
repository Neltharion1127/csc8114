"""
plot_server_metrics.py
======================
Reads the server_log_<session>.csv and produces a multi-panel
dashboard covering:

  Panel 1 — Loss over rounds          (train vs test, per client)
  Panel 2 — Rain classification acc   (rain_correct per round)
  Panel 3 — Latency breakdown         (decomp + comp time per round)
  Panel 4 — Gradient magnitude        (per round, train only)

Usage:
    uv run python src/data/plot_server_metrics.py
    uv run python src/data/plot_server_metrics.py --log results/server_log_20260312150555.csv
"""
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── paths ────────────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def _find_latest_log() -> Path:
    # Use rglob to find all logs recursively inside results/
    import re
    all_csvs = list((project_root / "results").rglob("server_log_*.csv"))
    
    # Filter for logs that have a 14-digit timestamp in their name or parent folder
    # Then sort by the filename (which contains the timestamp)
    logs = sorted(all_csvs, key=lambda p: p.name)
    
    if not logs:
        raise FileNotFoundError(
            f"No server_log_*.csv found in {project_root / 'results'}"
        )
    
    latest = logs[-1]
    print(f"[DEBUG] Picking latest log: {latest.relative_to(project_root)}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────

def plot_server_metrics(log_path: Path):
    df = pd.read_csv(log_path)
    print(f"[INFO] Loaded {len(df):,} rows from {log_path.name}")

    # ── Compatibility: older logs without 'round' column ─────────────────────
    if "round" not in df.columns:
        # Bin rows into N equal segments so the chart stays readable
        N_BINS  = 20
        df      = df.copy()
        df["round"] = pd.cut(
            np.arange(len(df)), bins=N_BINS, labels=False
        )
        print(f"[WARN] 'round' column missing — binned {len(df):,} rows into {N_BINS} segments")

    if "rain_correct" not in df.columns:
        # Derive on-the-fly from target & prediction
        df["rain_correct"] = ((df["target"] > 0.1) == (df["prediction"] > 0.1)).astype(int)

    session_id = log_path.stem.replace("server_log_", "")

    # ── Split train / test ────────────────────────────────────────────────────
    df_train = df[df["is_training"] == 1] if "is_training" in df.columns else df
    df_test  = df[df["is_training"] == 0] if "is_training" in df.columns else pd.DataFrame()

    # ── Per-round aggregation helper ─────────────────────────────────────────
    def per_round(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return sub.groupby("round")[cols].mean().reset_index()

    palette = plt.cm.tab10.colors

    client_ids = sorted(df["client_id"].unique()) if "client_id" in df.columns else [0]

    # ─────────────────────────────────────────────────────────────────────────
    # Figure: 2 rows × 2 cols
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Server Metrics Dashboard — Session {session_id}",
        fontsize=14, fontweight="bold", y=1.01
    )

    # ── Panel 1: Loss over rounds (per client) ────────────────────────────────
    ax1 = axes[0, 0]
    for i, cid in enumerate(client_ids):
        color = palette[i % len(palette)]
        sub   = df_train[df_train["client_id"] == cid]
        if sub.empty:
            continue
        agg = per_round(sub, ["loss"])
        ax1.plot(agg["round"], agg["loss"], "o-",
                 color=color, linewidth=1.8, markersize=4,
                 label=f"Client {int(cid)} (train)")
        if not df_test.empty:
            sub_t = df_test[df_test["client_id"] == cid]
            if not sub_t.empty:
                agg_t = per_round(sub_t, ["loss"])
                ax1.plot(agg_t["round"], agg_t["loss"], "s--",
                         color=color, linewidth=1.2, markersize=3, alpha=0.6,
                         label=f"Client {int(cid)} (test)")
    ax1.set_title("📉 Loss per Round", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round");  ax1.set_ylabel("MSE Loss")
    ax1.legend(fontsize=7);  ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_facecolor("#f8f8f8")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Panel 2: Accuracy (rain classification) ───────────────────────────────
    ax2 = axes[0, 1]
    for i, cid in enumerate(client_ids):
        color = palette[i % len(palette)]
        sub   = df_train[df_train["client_id"] == cid]
        if sub.empty:
            continue
        agg = per_round(sub, ["rain_correct"])
        ax2.plot(agg["round"], agg["rain_correct"] * 100, "o-",
                 color=color, linewidth=1.8, markersize=4,
                 label=f"Client {int(cid)}")

    # Rain-only accuracy
    rain_df = df_train[df_train["rain_flag"] == 1] if "rain_flag" in df.columns else pd.DataFrame()
    if not rain_df.empty:
        agg_rain = per_round(rain_df, ["rain_correct"])
        ax2.plot(agg_rain["round"], agg_rain["rain_correct"] * 100, "k^--",
                 linewidth=1.2, markersize=4, alpha=0.5, label="🌧 Rain only")

    ax2.axhline(50, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Random baseline")
    ax2.set_title("☔ Rain Classification Accuracy", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round");  ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=7);  ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_facecolor("#f8f8f8")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Panel 3: Latency (decomp + compute) per round ─────────────────────────
    ax3 = axes[1, 0]
    agg_latency = per_round(df_train, ["decompression_time_ms", "computation_time_ms"])
    x = agg_latency["round"].values
    decomp  = agg_latency["decompression_time_ms"].values
    compute = agg_latency["computation_time_ms"].values

    ax3.bar(x, decomp,  label="Decomp (ms)", color="#5b8dee", alpha=0.8, width=0.4)
    ax3.bar(x, compute, bottom=decomp,
            label="Compute (ms)", color="#f9a825", alpha=0.8, width=0.4)

    total_avg = (decomp + compute).mean()
    ax3.axhline(total_avg, color="red", linestyle="--", linewidth=1.2,
                label=f"Avg total = {total_avg:.2f} ms")
    ax3.set_title("⚡ Latency per Round (Train)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Round");  ax3.set_ylabel("Time (ms)")
    ax3.legend(fontsize=7);  ax3.grid(True, alpha=0.2, axis="y", linestyle="--")
    ax3.set_facecolor("#f8f8f8")
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── Panel 4: Gradient magnitude per round (train only) ────────────────────
    ax4 = axes[1, 1]
    agg_grad = per_round(df_train, ["gradient_magnitude"])
    ax4.plot(agg_grad["round"], agg_grad["gradient_magnitude"], "D-",
             color="#7b2d8b", linewidth=1.8, markersize=4)
    ax4.fill_between(agg_grad["round"], 0, agg_grad["gradient_magnitude"],
                     color="#7b2d8b", alpha=0.1)

    grad_mean = agg_grad["gradient_magnitude"].mean()
    ax4.axhline(grad_mean, color="grey", linestyle="--", linewidth=1,
                label=f"Mean = {grad_mean:.4f}")

    # Warn if many zeros
    zero_pct = (df_train["gradient_magnitude"] == 0).mean() * 100
    if zero_pct > 20:
        ax4.text(0.5, 0.85, f"⚠️ {zero_pct:.0f}% zeros (test passes included?)",
                 transform=ax4.transAxes, ha="center", fontsize=8, color="red")

    ax4.set_title("∇ Gradient Magnitude per Round (Train)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Round");  ax4.set_ylabel("||grad||")
    ax4.legend(fontsize=7);  ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.set_facecolor("#f8f8f8")
    ax4.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()

    out_dir  = project_root / "results" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"server_metrics_{session_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n✅ Dashboard saved: {out_path}")

    # ── Text summary ─────────────────────────────────────────────────────────
    print("\n── Server Metrics Summary ────────────────────────────────────────")
    print(f"  Total log entries : {len(df):,}  "
          f"(train={len(df_train):,}, test={len(df_test):,})")
    if not df_train.empty:
        avg_loss    = df_train["loss"].mean()
        avg_acc     = df_train["rain_correct"].mean() * 100
        avg_decomp  = df_train["decompression_time_ms"].mean()
        avg_compute = df_train["computation_time_ms"].mean()
        avg_grad    = df_train["gradient_magnitude"].mean()
        print(f"  Avg Train Loss    : {avg_loss:.4f}")
        print(f"  Avg Accuracy      : {avg_acc:.2f}%")
        print(f"  Avg Latency       : decomp={avg_decomp:.2f} ms  "
              f"compute={avg_compute:.2f} ms  "
              f"total={avg_decomp+avg_compute:.2f} ms")
        print(f"  Avg Gradient Mag  : {avg_grad:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path to server_log_*.csv. Defaults to the latest one in results/"
    )
    args = parser.parse_args()

    log_path = Path(args.log) if args.log else _find_latest_log()
    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        sys.exit(1)

    plot_server_metrics(log_path)


if __name__ == "__main__":
    main()
