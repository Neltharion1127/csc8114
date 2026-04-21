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
    "font.family":    "serif",
    "font.serif":     ["Times New Roman", "DejaVu Serif"],
    "font.size":      9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
SESSION     = "2026-04-21_00-51-00"
SCENARIO    = "09"
OUT_PDF     = RESULTS_DIR / "graphics" / "figA_scheduler_timeline.pdf"
OUT_PNG     = RESULTS_DIR / "graphics" / "figA_scheduler_timeline.png"

# Clients to display and their latency profile descriptions
CLIENTS = [
    (1,  "Client 1 (~1.5 ms)"),
    (5,  "Client 5 (~5.5 ms)"),
    (8,  "Client 8 (~10.5 ms)"),
]

# Scheduler thresholds (must match config.yaml → scheduler section)
FLOAT16_THRESHOLD = 4.0
INT8_THRESHOLD    = 10.0
EMA_ALPHA         = 0.2

# Compression mode colours (Wong 2011) and ordering
MODE_COLORS = {
    "float32":  "#0072B2",   # blue
    "float16":  "#009E73",   # green
    "int8":     "#D55E00",   # vermillion
    "topk_int8":"#CC79A7",   # pink
}
MODE_ORDER = ["float32", "float16", "int8", "topk_int8"]


# --- Data loading -----------------------------------------------------------

def load_client(client_id: int) -> pd.DataFrame:
    """
    Load per-step TRAIN rows for one client.
    Assigns a monotonic global step index and computes the EMA latency.
    """
    pattern = f"training_log_client{client_id}_2*.csv"
    files = sorted((RESULTS_DIR / SESSION / SCENARIO).glob(pattern))
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
    fig, axes = plt.subplots(n_clients, 1, figsize=(3.5, 2.6 * n_clients),
                             sharex=False)
    if n_clients == 1:
        axes = [axes]

    for ax, (cid, title) in zip(axes, CLIENTS):
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

        # --- colour strip at the bottom (y < 0) ----------------------------
        strip_h = ema.max() * 0.06   # height = 6% of y-range
        for i, (s, mode) in enumerate(zip(steps, modes)):
            color = MODE_COLORS.get(str(mode), "#888888")
            ax.bar(s, strip_h, bottom=-strip_h, width=1.0,
                   color=color, align="center", linewidth=0, zorder=2)

        # --- EMA latency line -----------------------------------------------
        ax.plot(steps, ema, color="#444444", linewidth=0.9, zorder=3,
                label="EMA latency")

        # --- threshold lines ------------------------------------------------
        ax.axhline(FLOAT16_THRESHOLD, color=MODE_COLORS["float16"],
                   linewidth=0.8, linestyle="--", zorder=1,
                   label=f"float16 thr. ({FLOAT16_THRESHOLD} ms)")
        ax.axhline(INT8_THRESHOLD, color=MODE_COLORS["int8"],
                   linewidth=0.8, linestyle="--", zorder=1,
                   label=f"int8 thr. ({INT8_THRESHOLD} ms)")

        ax.set_title(title, fontsize=8)
        ax.set_ylabel("EMA Latency (ms)")
        ax.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        ymax = max(ema.max(), INT8_THRESHOLD) * 1.15
        ax.set_ylim(-strip_h, ymax)
        ax.set_yticks([0, FLOAT16_THRESHOLD, INT8_THRESHOLD,
                       int(round(ymax / 5) * 5)])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel("Training Step")

    # --- shared legend (modes + latency line) -------------------------------
    mode_patches = [
        mpatches.Patch(facecolor=MODE_COLORS[m], label=m)
        for m in MODE_ORDER
        if m in MODE_COLORS
    ]
    ema_line = plt.Line2D([0], [0], color="#444444", linewidth=0.9,
                          label="EMA latency")
    axes[0].legend(
        handles=mode_patches + [ema_line],
        loc="upper right",
        frameon=True, framealpha=0.9, edgecolor="#cccccc",
        fontsize=6.5, borderpad=0.4, labelspacing=0.2,
        title="Mode (strip)", title_fontsize=6.5,
    )

    fig.tight_layout(pad=0.5, h_pad=0.8)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    draw()
