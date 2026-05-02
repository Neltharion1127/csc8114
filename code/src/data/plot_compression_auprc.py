"""
Figure 2: Effect of compression mode on AUPRC, grouped by latency condition.

Purpose
-------
Shows that activation compression (float16 / int8 / Adaptive topk_int8) incurs
negligible accuracy loss relative to the float32 baseline, while substantially
reducing per-step communication payload. Supports the paper's claim that the
adaptive scheduler achieves a favourable accuracy–bandwidth trade-off.

Data source
-----------
- results/matrix_summary.csv  (one row per scenario × seed run)
- Columns used: scenario_id, auprc_mean, avg_payload_bytes
- Three seeds (42, 52, 62) per scenario; error bars = ±1 std across seeds.
- Payload labels in the legend are read from avg_payload_bytes at runtime so
  they update automatically when M09/M14 are re-run after the topk_int8 fix.

Scenarios included
------------------
  No latency : M01 (float32), M02 (float16), M03 (int8)
               — no Adaptive scenario at zero latency (scheduler stays at float32)
  Mid 10 ms  : M05 (float32), M06 (float16), M07 (int8), M09 (Adaptive)
  High 63 ms : M10 (float32), M11 (float16), M12 (int8), M14 (Adaptive)

Scenarios excluded
------------------
  M04, M08, M13  (ρ=3 sync-interval ablation — different experimental axis)

Notes
-----
- Adaptive payload label is computed per latency group from avg_payload_bytes,
  so mid (~92 B) and high (~52 B post topk_int8 fix) are reported separately.
- Dotted horizontal lines mark the float32 baseline AUPRC within each group.
- Figure sized for a single column (3.5 × 2.6 in) in a two-column paper.

Output
------
  results/graphics/fig2_compression_auprc.pdf  (vector, for LaTeX)
  results/graphics/fig2_compression_auprc.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_compression_auprc.py

LaTeX inclusion
---------------
  \\begin{figure}
    \\includegraphics[width=\\linewidth]{fig2_compression_auprc.pdf}
    \\caption{Mean AUPRC (±std, 3 seeds) per compression mode and latency
              condition. Dotted lines mark the float32 baseline within each
              group. Payload sizes are averaged across all clients and rounds.}
  \\end{figure}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# --- LaTeX-compatible style -------------------------------------------------
plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":             9,
    "axes.titlesize":        10,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8,
    "lines.linewidth":       1.0,
    "axes.linewidth":        0.7,
    "xtick.major.width":     0.7,
    "ytick.major.width":     0.7,
    "pdf.fonttype":          42,   # embed fonts as TrueType (Acrobat-safe)
    "ps.fonttype":           42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUT_PDF = RESULTS_DIR / "graphics" / "fig2_compression_auprc.pdf"
OUT_PNG = RESULTS_DIR / "graphics" / "fig2_compression_auprc.png"

# --- Scenario metadata ------------------------------------------------------
# ρ=3 scenarios (M04, M08, M13) excluded — different experimental axis.
# Adaptive is only present at Mid and High latency (scheduler stays at float32
# when there is no network pressure).
SCENARIO_META = {
    "N01": ("No latency",   "float32"),
    "N02": ("No latency",   "float16"),
    "N03": ("No latency",   "int8"),
    "L05": ("Low (~8 ms)",  "float32"),
    "L06": ("Low (~8 ms)",  "float16"),
    "L07": ("Low (~8 ms)",  "int8"),
    "L09": ("Low (~8 ms)",  "Adaptive"),
    "H11": ("High (~50 ms)", "float32"),
    "H12": ("High (~50 ms)", "float16"),
    "H13": ("High (~50 ms)", "int8"),
    "H15": ("High (~50 ms)", "Adaptive"),
}

LATENCY_ORDER     = ["No latency", "Low (~8 ms)", "High (~50 ms)"]
COMPRESSION_ORDER = ["float32", "float16", "int8", "Adaptive"]

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "float32":  "#0072B2",   # blue
    "float16":  "#009E73",   # green
    "int8":     "#D55E00",   # vermillion
    "Adaptive": "#CC79A7",   # pink
}


# --- Data loading -----------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "matrix_summary.csv", dtype={"scenario_id": str})
    df = df[df["scenario_id"].isin(SCENARIO_META)]
    df["latency_group"] = df["scenario_id"].map(lambda s: SCENARIO_META[s][0])
    df["compression"]   = df["scenario_id"].map(lambda s: SCENARIO_META[s][1])
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean/std AUPRC and mean payload across seeds."""
    auprc = (
        df.groupby(["latency_group", "compression"])["auprc_mean"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    payload = (
        df.groupby(["latency_group", "compression"])["avg_payload_bytes"]
        .mean()
        .reset_index()
        .rename(columns={"avg_payload_bytes": "payload_b"})
    )
    stats = auprc.merge(payload, on=["latency_group", "compression"])
    stats["std"] = stats["std"].fillna(0.0)
    return stats


def _payload_label(comp: str, stats: pd.DataFrame) -> str:
    """
    Build legend label with dynamic payload info read from data.
    For Adaptive, reports mid and high payloads separately since they differ
    (mid ~92 B with topk_int8 fix; high ~52 B).
    For fixed modes, payload is the same across latency groups.
    """
    if comp == "float32":
        row = stats[stats["compression"] == comp].iloc[0]
        return f"float32 (baseline, {row['payload_b']:.0f} B)"
    if comp == "Adaptive":
        low  = stats[(stats["compression"] == comp) & (stats["latency_group"] == "Low (~8 ms)")]
        high = stats[(stats["compression"] == comp) & (stats["latency_group"] == "High (~50 ms)")]
        parts = []
        if not low.empty:
            parts.append(f"low {low.iloc[0]['payload_b']:.0f} B")
        if not high.empty:
            parts.append(f"high {high.iloc[0]['payload_b']:.0f} B")
        return f"Adaptive ({' / '.join(parts)})"
    # float16 / int8: payload is latency-independent
    rows = stats[stats["compression"] == comp]
    if rows.empty:
        return comp
    p = rows.iloc[0]["payload_b"]
    baseline = stats[(stats["compression"] == "float32") &
                     (stats["latency_group"] == rows.iloc[0]["latency_group"])]
    if not baseline.empty and baseline.iloc[0]["payload_b"] > 0:
        pct = int(round((1 - p / baseline.iloc[0]["payload_b"]) * 100))
        return f"{comp} (−{pct}%, {p:.0f} B)"
    return f"{comp} ({p:.0f} B)"


# --- Drawing ----------------------------------------------------------------

def draw(stats: pd.DataFrame) -> None:
    # Single-column width for a two-column paper
    fig_w, fig_h = 3.5, 2.6
    bar_w     = 0.16
    group_gap = 0.85

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    n_lat  = len(LATENCY_ORDER)
    n_comp = len(COMPRESSION_ORDER)
    x_centres = np.arange(n_lat, dtype=float) * group_gap
    offsets   = np.linspace(-(n_comp - 1) / 2, (n_comp - 1) / 2, n_comp) * bar_w

    for ci, comp in enumerate(COMPRESSION_ORDER):
        sub = stats[stats["compression"] == comp].set_index("latency_group")
        xs, means, stds = [], [], []
        for li, lat in enumerate(LATENCY_ORDER):
            if lat in sub.index:
                xs.append(x_centres[li] + offsets[ci])
                means.append(sub.loc[lat, "mean"])
                stds.append(sub.loc[lat, "std"])

        ax.bar(
            xs, means,
            width=bar_w * 0.95,
            color=COLORS[comp],
            label=comp,
            yerr=stds,
            capsize=2,
            error_kw={"elinewidth": 0.8, "ecolor": "#444444", "capthick": 0.8},
            zorder=3,
        )

    # Dotted baseline reference line per latency group (float32 mean)
    baseline = stats[stats["compression"] == "float32"].set_index("latency_group")
    for li, lat in enumerate(LATENCY_ORDER):
        if lat in baseline.index:
            bval  = baseline.loc[lat, "mean"]
            left  = x_centres[li] - group_gap / 2 + 0.04
            right = x_centres[li] + group_gap / 2 - 0.04
            ax.hlines(bval, left, right, colors="#666666", linestyles=":",
                      linewidth=0.8, zorder=2)

    ax.set_xticks(x_centres)
    ax.set_xticklabels(LATENCY_ORDER)
    ax.set_xlabel("Network Latency Condition")
    ax.set_ylabel("AUPRC")

    ymin = stats["mean"].min() - 0.018
    ymax = stats["mean"].max() + 0.012
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.002))
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=COLORS[c], label=_payload_label(c, stats))
        for c in COMPRESSION_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        handlelength=1.2,
        handleheight=0.9,
        borderpad=0.5,
        labelspacing=0.3,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    df    = load_data()
    stats = compute_stats(df)
    print(stats.to_string(index=False))
    draw(stats)
