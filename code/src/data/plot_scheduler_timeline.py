"""
Figure A: Adaptive scheduler timeline — per-step latency and compression mode.

Purpose
-------
Illustrates how the adaptive scheduler reacts to measured network latency in
real time.  The EMA-smoothed latency trace is plotted alongside horizontal
threshold lines, and the bottom colour strip shows the resulting compression
mode assigned to each forward step.  Three representative clients are shown
to capture heterogeneous behaviour arising from their different simulated
latency offsets:
  - Client 1 (~1.5 ms base): mostly float16
  - Client 5 (~5.5 ms base): mixed float16/int8
  - Client 8 (~10.5 ms base): mostly int8

Data source
-----------
- results/2026-04-21_00-51-00/09/training_log_client<N>_*.csv
  M09: Mid latency (10 ms), Adaptive scheduler, seed=42 (single seed shown).
- Per-step raw LatencyMs and NextCompression are logged by the client.
- EMA is recomputed here from raw values with the same alpha=0.2 used at
  runtime so the smoothed curve matches the scheduler's internal state.

Output
------
  results/graphics/figA_scheduler_timeline.pdf  (vector, for LaTeX)
  results/graphics/figA_scheduler_timeline.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_scheduler_timeline.py

LaTeX inclusion
---------------
  \\begin{figure}
    \\includegraphics[width=\\linewidth]{figA_scheduler_timeline.pdf}
    \\caption{Adaptive scheduler behaviour across training steps for three
              clients with different simulated latency offsets (Mid 10 ms
              scenario, seed=42).  Lines show EMA-smoothed latency; the
              colour strip at the bottom of each panel indicates the
              compression mode assigned by the scheduler.  Dashed horizontal
              lines mark the float16 (4 ms) and int8 (10 ms) thresholds.}
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
SCENARIO    = "L09_seed42"
OUT_PDF     = RESULTS_DIR / "graphics" / "figA_scheduler_timeline.pdf"
OUT_PNG     = RESULTS_DIR / "graphics" / "figA_scheduler_timeline.png"

# Clients to display and their latency profile descriptions
CLIENTS = [
    (1,  "Client 1 (mostly float16)"),
    (5,  "Client 5 (float16/int8)"),
    (11, "Client 11 (mostly int8)"),
]

# Scheduler thresholds (must match config.yaml → scheduler section)
FLOAT16_THRESHOLD = 4.0
INT8_THRESHOLD    = 10.0
EMA_ALPHA         = 0.2

# Compression mode colours (Wong 2011) and ordering
MODE_COLORS = {
    "float32": "#40A9FF",   # blue
    "float16": "#45DAD1",   # teal
    "int8":    "#FFA940",   # orange
}
MODE_ORDER = ["float32", "float16", "int8"]


# --- Data loading -----------------------------------------------------------

def load_client(client_id: int) -> pd.DataFrame:
    """
    Load per-step TRAIN rows for one client.
    Assigns a monotonic global step index and computes the EMA latency.
    """
    pattern = f"training_log_client{client_id}_2*.csv"
    files = sorted((RESULTS_DIR / SESSION / SCENARIO).glob(pattern))
    files = [f for f in files if "_meta" not in f.name]
    if not files:
        raise FileNotFoundError(f"No log for client {client_id} in {SESSION}/{SCENARIO}")
    df = pd.read_csv(files[0])
    train = df[df["Status"] == "TRAIN"].copy().reset_index(drop=True)
    train["GlobalStep"] = np.arange(1, len(train) + 1)

    # Recompute EMA to get a smooth curve independent of per-row noise
    ema = []
    state = 0.0
    for lat in train["LatencyMs"]:
        if state <= 0.0:
            state = lat
        else:
            state = EMA_ALPHA * lat + (1.0 - EMA_ALPHA) * state
        ema.append(state)
    train["EMA"] = ema
    return train


# --- Drawing ----------------------------------------------------------------

def draw() -> None:
    n_clients = len(CLIENTS)
    # Horizontal layout: 1 row × n_clients columns — matches fig5 width
    fig, axes = plt.subplots(1, n_clients, figsize=(7.0, 2.4), sharey=False)

    for col, (ax, (cid, title)) in enumerate(zip(axes, CLIENTS)):
        try:
            df = load_client(cid)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(title)
            continue

        steps = df["GlobalStep"].values
        ema   = df["EMA"].values
        modes = df["NextCompression"].values

        # colour strip at the bottom
        strip_h = max(ema.max(), INT8_THRESHOLD) * 0.07
        for s, mode in zip(steps, modes):
            color = MODE_COLORS.get(str(mode), "#888888")
            ax.bar(s, strip_h, bottom=-strip_h, width=1.0,
                   color=color, align="center", linewidth=0, zorder=2)

        # EMA latency line
        ax.plot(steps, ema, color="#333333", linewidth=1.0, zorder=3)

        # threshold lines
        ax.axhline(FLOAT16_THRESHOLD, color=MODE_COLORS["float16"],
                   linewidth=0.9, linestyle="--", zorder=1,
                   label=f"float16 ({FLOAT16_THRESHOLD} ms)")
        ax.axhline(INT8_THRESHOLD, color=MODE_COLORS["int8"],
                   linewidth=0.9, linestyle="--", zorder=1,
                   label=f"int8 ({INT8_THRESHOLD} ms)")

        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Training Step")
        ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ymax = max(ema.max(), INT8_THRESHOLD) * 1.18
        ax.set_ylim(-strip_h, ymax)
        ax.set_yticks([0, FLOAT16_THRESHOLD, INT8_THRESHOLD,
                       round(ymax / 5) * 5])
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        if col == 0:
            ax.set_ylabel("EMA Latency (ms)")

    # Legend on rightmost panel
    mode_patches = [
        mpatches.Patch(facecolor=MODE_COLORS[m], label=m)
        for m in MODE_ORDER
    ]
    ema_handle = plt.Line2D([0], [0], color="#333333", linewidth=1.0,
                            label="EMA latency")
    axes[-1].legend(
        handles=mode_patches + [ema_handle],
        loc="upper right",
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        borderpad=0.4, labelspacing=0.25,
        title="Mode (strip)", title_fontsize=7.5,
    )

    fig.tight_layout(pad=0.5, w_pad=1.0)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    draw()
