"""
Figure 4: Cross-platform AUPRC comparison — simulation vs Raspberry Pi.

Purpose
-------
Shows that simulation AUPRC rankings and values closely match those measured
on physical Raspberry Pi hardware, validating the simulation methodology.
Four communication strategies are compared: float32 ρ=1 (baseline),
float16 ρ=1, float32 ρ=3, and joint adaptive.

Data sources
------------
- Simulation: eval report JSONs for N01, N02, N04, H16 (paired scenarios).
- Pi: hardcoded from physical deployment results (3 seeds each, mean ± std).

Pairing rationale
-----------------
  float32 ρ=1   : P1  ↔  N01
  float16 ρ=1   : P2  ↔  N02
  float32 ρ=3   : P3  ↔  N04
  joint adaptive : P4  ↔  H16  (both converge to int8 + ρ=3)

Output
------
  results/graphics/fig4_cross_platform.pdf  (vector, for LaTeX)
  results/graphics/fig4_cross_platform.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_cross_platform.py
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import statistics as _st
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
OUT_PDF = RESULTS_DIR / "graphics" / "fig4_cross_platform.pdf"
OUT_PNG = RESULTS_DIR / "graphics" / "fig4_cross_platform.png"

# --- Strategy definitions ---------------------------------------------------
STRATEGIES = [
    ("float32, ρ=1\n(baseline)", "N01"),
    ("float16, ρ=1",             "N02"),
    ("float32, ρ=3",             "N04"),
    ("Joint\nAdaptive",          "H16"),
]

# Pi AUPRC values: (mean, std) from physical deployment, 3 seeds each
PI_AUPRC = {
    "N01": (0.6381, 0.0042),
    "N02": (0.6434, 0.0044),
    "N04": (0.6479, 0.0006),
    "H16": (0.6482, 0.0011),
}

# Colorblind-safe palette (Wong 2011)
COLOR_SIM = "#6682F5"   # periwinkle (simulation)
COLOR_PI  = "#F57582"   # coral (real hardware)


# --- Data loading -----------------------------------------------------------

def load_sim_auprc(scenario_id: str) -> tuple[float, float]:
    """Load mean and std AUPRC across seeds from eval report JSONs."""
    values = []
    for seed in SEEDS:
        f = RESULTS_DIR / SESSION / f"{scenario_id}_seed{seed}_eval_report.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        values.append(d["weighted_overall"]["auprc"])
    if not values:
        raise FileNotFoundError(f"No eval reports found for {scenario_id}")
    mean = _st.mean(values)
    std  = _st.stdev(values) if len(values) > 1 else 0.0
    return mean, std


# --- Drawing ----------------------------------------------------------------

def draw() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    n = len(STRATEGIES)
    x = np.arange(n, dtype=float)
    bar_w = 0.32

    sim_means, sim_stds = [], []
    pi_means,  pi_stds  = [], []

    for label, sid in STRATEGIES:
        sm, ss = load_sim_auprc(sid)
        pm, ps = PI_AUPRC[sid]
        sim_means.append(sm); sim_stds.append(ss)
        pi_means.append(pm);  pi_stds.append(ps)
        print(f"  {sid}: sim={sm:.4f}±{ss:.4f}  pi={pm:.4f}±{ps:.4f}")

    ax.bar(x - bar_w / 2, sim_means, width=bar_w, color=COLOR_SIM,
           label="Simulation", yerr=sim_stds, capsize=2,
           error_kw={"elinewidth": 0.8, "ecolor": "#444", "capthick": 0.8},
           zorder=3)
    ax.bar(x + bar_w / 2, pi_means, width=bar_w, color=COLOR_PI,
           label="Raspberry Pi", yerr=pi_stds, capsize=2,
           error_kw={"elinewidth": 0.8, "ecolor": "#444", "capthick": 0.8},
           zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _ in STRATEGIES], fontsize=7.5)
    ax.set_ylabel("AUPRC")

    all_means = sim_means + pi_means
    all_stds  = sim_stds  + pi_stds
    ymin = min(m - s for m, s in zip(all_means, all_stds)) - 0.003
    ymax = max(m + s for m, s in zip(all_means, all_stds)) + 0.003
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=COLOR_SIM, label="Simulation"),
        mpatches.Patch(facecolor=COLOR_PI,  label="Raspberry Pi"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="#cccccc",
              handlelength=1.0, borderpad=0.4, labelspacing=0.2)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    draw()
