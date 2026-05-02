"""
Figure 5: Effect of sync interval rho on convergence (validation AUPRC per round).

Purpose
-------
Compares ρ=1 (S0/S0 baseline, sync every step) vs ρ=3 (S3, sync every 3 steps)
across three simulated latency conditions (No / Mid 10 ms / High 63 ms).
Shows that reducing sync frequency consistently lowers peak AUPRC and causes
earlier early-stopping, supporting the paper's claim that frequent aggregation
is critical for federated convergence.

Data source
-----------
- training_log_client1_<timestamp>.csv  (one file per seed run, client 1 only)
  Each file has columns: Epoch, Status (TRAIN/VAL), RainFlag, RainProbability, ...
- "Epoch" counts local training steps. For ρ=3, sync occurs every 3 epochs, so
  federation_round = epoch / rho.
- AUPRC is computed per epoch from VAL rows using sklearn.average_precision_score.
  NOTE: this is client-side *validation* AUPRC on the local sensor split, not the
  global test-set AUPRC reported in the main results table. Values (~0.35–0.52)
  are lower than test-set values (~0.70) due to the per-sensor data distribution.
- Three seeds (42, 52, 62) per scenario; shaded band = ±1 std across seeds.
- Only rounds where all three seeds have data are plotted (avoids noisy tail
  when seeds stop at different rounds due to early stopping).

Scenarios used
--------------
  M01 (ρ=1, no lat)   vs  M04 (ρ=3, no lat)
  M05 (ρ=1, mid lat)  vs  M08 (ρ=3, mid lat)
  M10 (ρ=1, high lat) vs  M13 (ρ=3, high lat)
  (M02, M03, M06, M07, M09, M11, M12, M14 are excluded — different compression axis)

Output
------
  results/graphics/fig5_rho_convergence.pdf  (vector, for LaTeX)
  results/graphics/fig5_rho_convergence.png  (raster preview, dpi=200)

Usage
-----
  uv run python src/data/plot_rho_convergence.py

LaTeX inclusion
---------------
  \\begin{figure}
    \\includegraphics[width=\\linewidth]{fig5_rho_convergence.pdf}
    \\caption{Validation AUPRC per federation round for sync intervals
              $\\rho=1$ (solid) and $\\rho=3$ (dashed). Shaded bands show
              $\\pm 1$ std across three random seeds. Curves end when all
              three seeds have triggered early stopping.}
  \\end{figure}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

# --- LaTeX-compatible style (serif font, embeds TrueType) -------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.2,
    "axes.linewidth":     0.7,
    "pdf.fonttype":       42,   # embed fonts as TrueType (Acrobat-safe)
    "ps.fonttype":        42,
})

# --- Paths ------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
# All six scenarios live in the same matrix session
SESSION = "2026-04-30_01-17-30"
OUT_PDF = RESULTS_DIR / "graphics" / "fig5_rho_convergence.pdf"
OUT_PNG = RESULTS_DIR / "graphics" / "fig5_rho_convergence.png"

# --- Experiment pairs -------------------------------------------------------
# (scenario_id, rho, latency_group_label)
# Only include the baseline (float32) scenarios to isolate the rho effect.
PAIRS = [
    ("N01", 1, "No latency"),
    ("N04", 3, "No latency"),
    ("L05", 1, "Low (~8 ms)"),
    ("L08", 3, "Low (~8 ms)"),
    ("H11", 1, "High (~50 ms)"),
    ("H14", 3, "High (~50 ms)"),
]
LATENCY_GROUPS = ["No latency", "Low (~8 ms)", "High (~50 ms)"]

# Wong (2011) colorblind-safe palette, one colour per latency group
GROUP_COLORS = {
    "No latency":    "#0072B2",   # blue
    "Low (~8 ms)":   "#009E73",   # green
    "High (~50 ms)": "#D55E00",   # vermillion
}


# --- Data loading -----------------------------------------------------------

def auprc_per_epoch(log_path: Path) -> pd.Series:
    """
    Read one seed's training log and return validation AUPRC per epoch.
    Epochs with no positive or no negative labels are skipped (AUPRC undefined).
    """
    df = pd.read_csv(log_path)
    val = df[df["Status"] == "VAL"]
    results = {}
    for epoch, grp in val.groupby("Epoch"):
        y_true = grp["RainFlag"].values.astype(int)
        y_prob = grp["RainProbability"].values
        # Skip degenerate epochs (all-positive or all-negative labels)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        results[int(epoch)] = average_precision_score(y_true, y_prob)
    return pd.Series(results)


def load_scenario(scenario_id: str, rho: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load all seed training logs for a scenario and return per-round statistics.

    Each seed's local Epoch index is divided by rho to express progress as
    federation rounds (the unit that is comparable across rho values).
    Only rounds present in ALL seeds are retained so the std estimate is valid.

    Returns
    -------
    rounds : np.ndarray  — federation round indices
    mean   : np.ndarray  — mean AUPRC across seeds
    std    : np.ndarray  — std  AUPRC across seeds
    """
    scenario_dir = RESULTS_DIR / SESSION / scenario_id
    # Timestamped files = one per seed; the *_progress.csv is a merged copy, excluded
    log_files = sorted(scenario_dir.glob("training_log_client1_2*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No timestamped training logs in {scenario_dir}")

    seed_curves: list[pd.Series] = []
    for f in log_files:
        s = auprc_per_epoch(f)
        # Convert local epoch → federation round
        s.index = (s.index / rho).round().astype(int)
        seed_curves.append(s)

    pivot = pd.concat(seed_curves, axis=1).dropna()   # intersect rounds
    rounds = pivot.index.values
    mean   = pivot.mean(axis=1).values
    std    = pivot.std(axis=1).fillna(0).values

    max_r = int(rounds.max()) if len(rounds) else 0
    print(f"  {scenario_id} (rho={rho}): {len(log_files)} seeds, "
          f"common rounds 1–{max_r}")
    return rounds, mean, std


# --- Helpers ----------------------------------------------------------------

def smooth(arr: np.ndarray, w: int = 3) -> np.ndarray:
    """Moving-average smoothing to reduce step-to-step noise."""
    if w <= 1 or len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    padded = np.pad(arr, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


# --- Drawing ----------------------------------------------------------------

def draw() -> None:
    # Width = full two-column text width (7 in); height tuned for 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=False)

    for ax, lat in zip(axes, LATENCY_GROUPS):
        color = GROUP_COLORS[lat]

        for scenario_id, rho, lat_label in PAIRS:
            if lat_label != lat:
                continue
            try:
                rounds, mean, std = load_scenario(scenario_id, rho)
            except FileNotFoundError as e:
                print(f"  Skipping M{scenario_id}: {e}")
                continue

            ms = smooth(mean, w=3)
            # ρ=1 → solid, thicker;  ρ=3 → dashed, thinner
            ls = "-"  if rho == 1 else "--"
            lw = 1.4  if rho == 1 else 1.1
            ax.plot(rounds, ms, color=color, linestyle=ls, linewidth=lw,
                    label=f"$\\rho={rho}$")
            ax.fill_between(rounds, ms - std, ms + std,
                            color=color, alpha=0.15, linewidth=0)

        ax.set_title(lat)
        ax.set_xlabel("Federation Round")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)

    axes[0].set_ylabel("Validation AUPRC")

    fig.tight_layout(pad=0.6, w_pad=0.8)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"PDF → {OUT_PDF}")
    print(f"PNG → {OUT_PNG}")


if __name__ == "__main__":
    draw()
