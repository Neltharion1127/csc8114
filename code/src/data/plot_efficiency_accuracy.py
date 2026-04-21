"""
Figure 3: Accuracy–bandwidth trade-off scatter plot.

Purpose
-------
Shows how each compression strategy positions itself on the accuracy vs.
communication-cost plane. The adaptive scheduler (M09/M14) is expected to
sit near the Pareto frontier: lower payload than float32 with minimal AUPRC
loss. This is the primary visualisation of the paper's system contribution.

Data source
-----------
- results/matrix_summary.csv  (one row per scenario × seed)
- Columns used: scenario_id, auprc_mean, avg_payload_bytes
- Each plotted point = mean across 3 seeds (42, 52, 62).
- Error bars = ±1 std across seeds in both axes.
- No-latency scenarios (M01–M03) have payload=0 (profiler disabled) and are
  excluded from the scatter; their mean AUPRC is shown as a dashed reference
  line labelled "No-latency ceiling".

Scenarios plotted
-----------------
  Mid (10 ms) : M05 float32, M06 float16, M07 int8, M09 Adaptive
  High (63 ms): M10 float32, M11 float16, M12 int8, M14 Adaptive

Scenarios excluded
------------------
  M01–M03  (payload=0, profiler off — used only as AUPRC ceiling reference)
  M04, M08, M13  (ρ=3 sync-interval axis — different experimental dimension)

Note
----
M14 (High Adaptive) payload will drop from ~264 B to ~52 B after the
topk_int8 fix is re-run (make matrix ONLY=14). Re-run this script afterwards
to update the figure automatically.

Output
------
  results/graphics/fig3_efficiency_accuracy.pdf  (vector, for LaTeX)
  results/graphics/fig3_efficiency_accuracy.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_efficiency_accuracy.py

LaTeX inclusion
---------------
  \\begin{figure}
    \\includegraphics[width=\\linewidth]{fig3_efficiency_accuracy.pdf}
    \\caption{Accuracy--bandwidth trade-off for each compression strategy.
              Points show mean AUPRC and mean payload across three seeds;
              error bars show $\\pm 1$ std. The dashed line marks the
              AUPRC ceiling measured under zero network latency.}
  \\end{figure}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from pathlib import Path

# --- LaTeX-compatible style -------------------------------------------------
plt.rcParams.update({
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      9,
    "axes.labelsize": 9,
    "xtick.labelsize":8,
    "ytick.labelsize":8,
    "legend.fontsize":8,
    "axes.linewidth": 0.7,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUT_PDF = RESULTS_DIR / "graphics" / "fig3_efficiency_accuracy.pdf"
OUT_PNG = RESULTS_DIR / "graphics" / "fig3_efficiency_accuracy.png"

# --- Scenario metadata ------------------------------------------------------
# (latency_group, compression_label, point_label)
SCENARIO_META = {
    "05": ("Mid (10 ms)",  "float32",  "float32"),
    "06": ("Mid (10 ms)",  "float16",  "float16"),
    "07": ("Mid (10 ms)",  "int8",     "int8"),
    "09": ("Mid (10 ms)",  "Adaptive", "Adaptive"),
    "10": ("High (63 ms)", "float32",  "float32"),
    "11": ("High (63 ms)", "float16",  "float16"),
    "12": ("High (63 ms)", "int8",     "int8"),
    "14": ("High (63 ms)", "Adaptive", "Adaptive"),
}
# No-latency scenarios used only for the ceiling reference line
NO_LAT_IDS = {"01", "02", "03"}

# Colorblind-safe palette (Wong 2011)
GROUP_COLORS = {
    "Mid (10 ms)":  "#009E73",   # green
    "High (63 ms)": "#D55E00",   # vermillion
}

# Marker per compression mode
MARKERS = {
    "float32":  "o",
    "float16":  "s",
    "int8":     "^",
    "Adaptive": "*",
}
MARKER_SIZES = {
    "float32":  40,
    "float16":  40,
    "int8":     40,
    "Adaptive": 120,   # star is visually smaller, compensate
}

# Nudge labels to avoid overlap: (dx, dy) in data units
LABEL_OFFSET = {
    "05": ( 2, -0.0008),   # mid float32
    "06": ( 2,  0.0005),   # mid float16
    "07": (-2, -0.0010),   # mid int8
    "09": ( 2,  0.0005),   # mid adaptive
    "10": ( 2, -0.0008),   # high float32
    "11": ( 2,  0.0005),   # high float16
    "12": (-2, -0.0010),   # high int8
    "14": ( 2,  0.0005),   # high adaptive
}


# --- Data loading -----------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, float]:
    """
    Returns (scatter_df, ceiling_auprc).
    scatter_df has one row per scenario with mean/std of auprc and payload.
    ceiling_auprc is the mean AUPRC across no-latency scenarios (M01-M03).
    """
    df = pd.read_csv(RESULTS_DIR / "matrix_summary.csv", dtype={"scenario_id": str})
    df["scenario_id"] = df["scenario_id"].str.zfill(2)

    # No-latency ceiling
    ceiling = df[df["scenario_id"].isin(NO_LAT_IDS)]["auprc_mean"].mean()

    # Scatter points
    df = df[df["scenario_id"].isin(SCENARIO_META)]
    df["latency_group"] = df["scenario_id"].map(lambda s: SCENARIO_META[s][0])
    df["compression"]   = df["scenario_id"].map(lambda s: SCENARIO_META[s][1])

    stats = df.groupby("scenario_id").agg(
        latency_group=("latency_group", "first"),
        compression=("compression",   "first"),
        auprc_mean=("auprc_mean",     "mean"),
        auprc_std=("auprc_mean",      "std"),
        payload_mean=("avg_payload_bytes", "mean"),
        payload_std=("avg_payload_bytes",  "std"),
    ).reset_index()
    stats["auprc_std"]   = stats["auprc_std"].fillna(0)
    stats["payload_std"] = stats["payload_std"].fillna(0)
    return stats, ceiling


# --- Drawing ----------------------------------------------------------------

def draw(stats: pd.DataFrame, ceiling: float) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # No-latency AUPRC ceiling
    ax.axhline(ceiling, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(260, ceiling + 0.0004, "No-latency ceiling",
            fontsize=7, color="#555555", ha="right", va="bottom")

    for _, row in stats.iterrows():
        sid   = row["scenario_id"]
        color = GROUP_COLORS[row["latency_group"]]
        marker = MARKERS[row["compression"]]
        ms     = MARKER_SIZES[row["compression"]]

        ax.errorbar(
            row["payload_mean"], row["auprc_mean"],
            xerr=row["payload_std"],
            yerr=row["auprc_std"],
            fmt=marker,
            color=color,
            markersize=np.sqrt(ms),
            capsize=2,
            elinewidth=0.7,
            capthick=0.7,
            zorder=3,
        )

        # Point label
        dx, dy = LABEL_OFFSET.get(sid, (2, 0.0005))
        ax.annotate(
            row["compression"],
            xy=(row["payload_mean"], row["auprc_mean"]),
            xytext=(row["payload_mean"] + dx, row["auprc_mean"] + dy),
            fontsize=6.5,
            color=color,
            va="center",
        )

    ax.set_xlabel("Mean Payload per Step (bytes)")
    ax.set_ylabel("AUPRC")
    ax.set_xlim(40, 290)
    ymin = stats["auprc_mean"].min() - 0.005
    ymax = max(stats["auprc_mean"].max(), ceiling) + 0.005
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend: latency group (color) + compression mode (marker shape)
    color_handles = [
        mlines.Line2D([], [], color=c, marker="o", linestyle="none",
                      markersize=5, label=lat)
        for lat, c in GROUP_COLORS.items()
    ]
    shape_handles = [
        mlines.Line2D([], [], color="#888888", marker=m, linestyle="none",
                      markersize=5 if k != "Adaptive" else 8, label=k)
        for k, m in MARKERS.items()
    ]
    leg1 = ax.legend(handles=color_handles, loc="lower right",
                     title="Latency", frameon=True, framealpha=0.9,
                     edgecolor="#cccccc", fontsize=7, title_fontsize=7,
                     borderpad=0.4, labelspacing=0.2)
    ax.add_artist(leg1)
    ax.legend(handles=shape_handles, loc="upper left",
              title="Compression", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fontsize=7, title_fontsize=7,
              borderpad=0.4, labelspacing=0.2)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    stats, ceiling = load_data()
    print(f"No-latency ceiling AUPRC: {ceiling:.4f}")
    print(stats[["scenario_id", "compression", "latency_group",
                 "auprc_mean", "payload_mean"]].to_string(index=False))
    draw(stats, ceiling)
