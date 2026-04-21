"""
Figure B: Monthly prediction performance across the test period (Jul 2025 – Mar 2026).

Purpose
-------
Illustrates how model accuracy (F1) and regression error (MSE) vary by month on
the held-out test set. Highlights seasonal patterns: winter months (Nov–Feb) are
expected to have higher rainfall occurrence and thus more opportunity for the
model to demonstrate classification ability, while summer months may show lower
MSE but sparser rain events.

Data source
-----------
- results/<session>/evaluation_report_<session>.csv
  Column `monthly_details` holds a list of per-month dicts:
  {'Month': 'MM', 'MSE': float, 'Acc': float, 'F1': float, 'Samples': int}
  Months 07–12 are 2025 (beginning of test window); 01–03 are 2026.
- Values are averaged across all 11 clients (one row per client in the CSV).

Session used
------------
  High-latency adaptive (M14): results/2026-04-16_09-48-06/
  This scenario best illustrates real-world adaptive behaviour.

Output
------
  results/graphics/figB_monthly_performance.pdf  (vector, for LaTeX)
  results/graphics/figB_monthly_performance.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_monthly_performance.py

LaTeX inclusion
---------------
  \\begin{figure}
    \\includegraphics[width=\\linewidth]{figB_monthly_performance.pdf}
    \\caption{Monthly prediction performance on the held-out test set
              (Jul 2025 – Mar 2026), averaged across 11 clients.
              Grey shading marks winter months (Nov–Feb). Error bars
              show $\\pm 1$ std across clients.}
  \\end{figure}
"""

import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# --- LaTeX-compatible style -------------------------------------------------
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
SESSION     = "2026-04-16_09-48-06"
CSV_PATH    = RESULTS_DIR / SESSION / f"evaluation_report_{SESSION}.csv"
OUT_PDF     = RESULTS_DIR / "graphics" / "figB_monthly_performance.pdf"
OUT_PNG     = RESULTS_DIR / "graphics" / "figB_monthly_performance.png"

# Test period runs Jul 2025 → Mar 2026 (months 07–12 = 2025, 01–03 = 2026)
MONTH_ORDER  = ["07", "08", "09", "10", "11", "12", "01", "02", "03"]
MONTH_LABELS = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
WINTER_MONTHS = {"11", "12", "01", "02"}   # grey shading

BAR_COLOR_F1  = "#0072B2"   # blue  (Wong 2011)
BAR_COLOR_MSE = "#D55E00"   # vermillion (Wong 2011)


# --- Data loading -----------------------------------------------------------

def load_monthly() -> pd.DataFrame:
    """
    Parse monthly_details from each client row and return a tidy DataFrame
    with columns: month, F1, MSE, client_id.
    """
    df = pd.read_csv(CSV_PATH)
    records = []
    for _, row in df.iterrows():
        md = ast.literal_eval(row["monthly_details"])
        for entry in md:
            records.append({
                "client_id": row["client_id"],
                "month":     entry["Month"],
                "F1":        entry["F1"],
                "MSE":       entry["MSE"],
                "samples":   entry["Samples"],
            })
    return pd.DataFrame(records)


def aggregate(tidy: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across clients for each month."""
    stats = (
        tidy.groupby("month")
        .agg(
            F1_mean=("F1",  "mean"),
            F1_std=("F1",   "std"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE",  "std"),
            samples=("samples", "mean"),
        )
        .reindex(MONTH_ORDER)
        .reset_index()
    )
    stats["F1_std"]  = stats["F1_std"].fillna(0)
    stats["MSE_std"] = stats["MSE_std"].fillna(0)
    return stats


# --- Drawing ----------------------------------------------------------------

def draw(stats: pd.DataFrame) -> None:
    fig, (ax_f1, ax_mse) = plt.subplots(2, 1, figsize=(3.5, 3.8), sharex=True)

    x = np.arange(len(MONTH_ORDER))

    for ax in (ax_f1, ax_mse):
        for xi, m in enumerate(MONTH_ORDER):
            if m in WINTER_MONTHS:
                ax.axvspan(xi - 0.5, xi + 0.5, color="#e8e8e8", zorder=0)

    # --- F1 panel ---
    ax_f1.bar(
        x, stats["F1_mean"],
        width=0.6,
        color=BAR_COLOR_F1,
        yerr=stats["F1_std"],
        capsize=2,
        error_kw={"elinewidth": 0.8, "ecolor": "#444444", "capthick": 0.8},
        zorder=3,
    )
    ax_f1.set_ylabel("F1 Score")
    ax_f1.set_ylim(
        max(0, stats["F1_mean"].min() - stats["F1_std"].max() - 0.05),
        min(1, stats["F1_mean"].max() + stats["F1_std"].max() + 0.05),
    )
    ax_f1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_f1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax_f1.spines[["top", "right"]].set_visible(False)

    # --- MSE panel ---
    ax_mse.bar(
        x, stats["MSE_mean"],
        width=0.6,
        color=BAR_COLOR_MSE,
        yerr=stats["MSE_std"],
        capsize=2,
        error_kw={"elinewidth": 0.8, "ecolor": "#444444", "capthick": 0.8},
        zorder=3,
    )
    ax_mse.set_ylabel("MSE (mm²)")
    ax_mse.set_ylim(0, stats["MSE_mean"].max() + stats["MSE_std"].max() + 2)
    ax_mse.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax_mse.spines[["top", "right"]].set_visible(False)

    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels(MONTH_LABELS)
    ax_mse.set_xlabel("Month (Jul 2025 – Mar 2026)")

    # Shared legend: winter shading
    winter_patch = mpatches.Patch(facecolor="#e8e8e8", edgecolor="#cccccc",
                                  label="Winter (Nov–Feb)")
    ax_f1.legend(handles=[winter_patch], loc="lower right",
                 frameon=True, framealpha=0.9, edgecolor="#cccccc",
                 fontsize=7, borderpad=0.4)

    fig.tight_layout(pad=0.5, h_pad=0.6)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    tidy  = load_monthly()
    stats = aggregate(tidy)
    print(stats[["month", "F1_mean", "F1_std", "MSE_mean", "MSE_std",
                 "samples"]].to_string(index=False))
    draw(stats)
