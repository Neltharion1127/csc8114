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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

# --- LaTeX-compatible style -------------------------------------------------
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.linewidth":  0.7,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
SESSION     = "2026-05-10_01-43-06"
SEEDS       = [42, 52, 62]
OUT_PDF = RESULTS_DIR / "graphics" / "fig3_efficiency_accuracy.pdf"
OUT_PNG = RESULTS_DIR / "graphics" / "fig3_efficiency_accuracy.png"

# --- Scenario metadata ------------------------------------------------------
# (latency_group, compression_label, point_label)
SCENARIO_META = {
    "L05": ("Low (~8 ms)",   "float32",  "float32"),
    "L06": ("Low (~8 ms)",   "float16",  "float16"),
    "L07": ("Low (~8 ms)",   "int8",     "int8"),
    "L09": ("Low (~8 ms)",   "Adaptive", "Adaptive"),
    "H11": ("High (~50 ms)", "float32",  "float32"),
    "H12": ("High (~50 ms)", "float16",  "float16"),
    "H13": ("High (~50 ms)", "int8",     "int8"),
    "H15": ("High (~50 ms)", "Adaptive", "Adaptive"),
}
# No-latency scenarios used only for the ceiling reference line
NO_LAT_IDS = {"N01", "N02", "N03"}

# Colorblind-safe palette (Wong 2011)
GROUP_COLORS = {
    "Low (~8 ms)":   "#45DAD1",   # teal
    "High (~50 ms)": "#FFA940",   # orange
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
    "L05": ( 2, -0.0008),
    "L06": ( 2,  0.0005),
    "L07": (-2, -0.0010),
    "L09": ( 2,  0.0005),
    "H11": ( 2, -0.0008),
    "H12": ( 2,  0.0005),
    "H13": (-2, -0.0010),
    "H15": ( 2,  0.0005),
}


# --- Data loading -----------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, float]:
    """
    Returns (scatter_df, ceiling_auprc).
    scatter_df has one row per scenario with mean/std of auprc and payload.
    ceiling_auprc is the mean AUPRC across no-latency scenarios (N01-N03).
    """
    import json, statistics as _st

    all_rows = []
    for scenario_id in list(SCENARIO_META) + list(NO_LAT_IDS):
        for seed in SEEDS:
            f = RESULTS_DIR / SESSION / f"{scenario_id}_seed{seed}_eval_report.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            auprc = d["weighted_overall"]["auprc"]
            payload = _st.mean(
                float(c.get("payload_bytes") or 0) for c in d["clients"]
            )
            meta = SCENARIO_META.get(scenario_id, (None, None, None))
            all_rows.append({
                "scenario_id":      scenario_id,
                "seed":             seed,
                "auprc_mean":       auprc,
                "avg_payload_bytes": payload,
                "latency_group":    meta[0],
                "compression":      meta[1],
            })

    df = pd.DataFrame(all_rows)

    # No-latency ceiling
    ceiling = df[df["scenario_id"].isin(NO_LAT_IDS)]["auprc_mean"].mean()

    # Scatter points: aggregate per scenario across seeds
    scatter = df[df["scenario_id"].isin(SCENARIO_META)].copy()
    stats = scatter.groupby("scenario_id").agg(
        latency_group=("latency_group",    "first"),
        compression=("compression",        "first"),
        auprc_mean=("auprc_mean",          "mean"),
        auprc_std=("auprc_mean",           "std"),
        payload_mean=("avg_payload_bytes", "mean"),
        payload_std=("avg_payload_bytes",  "std"),
    ).reset_index()
    stats["auprc_std"]   = stats["auprc_std"].fillna(0)
    stats["payload_std"] = stats["payload_std"].fillna(0)
    return stats, ceiling


# --- Drawing ----------------------------------------------------------------

COMP_ORDER = ["float32", "float16", "int8", "Adaptive"]
# (latency_label, color, x_offset)
LAT_SERIES = [
    ("Low (~8 ms)",   "#45DAD1", -0.18),
    ("High (~50 ms)", "#FFA940", +0.18),
]


def draw(stats: pd.DataFrame, ceiling: float) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ymin = min(stats["auprc_mean"].min() - 0.006, ceiling - 0.003)
    ymax = max(stats["auprc_mean"].max(), ceiling) + 0.004
    ax.set_ylim(ymin, ymax)

    # Ceiling reference
    ax.axhline(ceiling, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(len(COMP_ORDER) - 0.3, ceiling + 0.0003, "No-latency ceiling",
            fontsize=7.5, color="#888888", ha="right", va="bottom")

    tick_labels = []
    for xi, comp in enumerate(COMP_ORDER):
        # Build x-tick label with payload(s)
        payloads = {}
        for lat, color, offset in LAT_SERIES:
            mask = (stats["compression"] == comp) & (stats["latency_group"] == lat)
            if mask.any():
                payloads[lat] = int(round(stats[mask]["payload_mean"].values[0]))
        vals = list(set(payloads.values()))
        if len(vals) == 1:
            tick_labels.append(f"{comp}\n({vals[0]} B)")
        else:
            lo = payloads.get("Low (~8 ms)", "—")
            hi = payloads.get("High (~50 ms)", "—")
            tick_labels.append(f"{comp}\n(L {lo} B / H {hi} B)")

        for lat, color, offset in LAT_SERIES:
            mask = (stats["compression"] == comp) & (stats["latency_group"] == lat)
            if not mask.any():
                continue
            row  = stats[mask].iloc[0]
            y    = row["auprc_mean"]
            yerr = row["auprc_std"]
            xpos = xi + offset

            ax.vlines(xpos, ymin, y, color=color, linewidth=2.0,
                      alpha=0.5, zorder=2)
            ax.errorbar(xpos, y, yerr=yerr, fmt="o", color=color,
                        markersize=8, capsize=3,
                        elinewidth=1.0, capthick=1.0, zorder=4)

    ax.set_xticks(np.arange(len(COMP_ORDER)))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=c, label=lat)
        for lat, c, _ in LAT_SERIES
    ]
    ax.legend(handles=handles, title="Latency", frameon=True, framealpha=0.9,
              edgecolor="#cccccc", fontsize=8, title_fontsize=8,
              borderpad=0.4, labelspacing=0.25, loc="lower right")

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
