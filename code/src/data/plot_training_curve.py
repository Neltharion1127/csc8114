"""
plot_training_curve.py
======================
Loads every periodic checkpoint (bestweights/<session>/periodic/)
and evaluates it on the 14-day test set.

Produces one figure with:
  - One subplot per client  : train_loss (from ckpt dict) vs test_mse (computed)
  - A combined overlay plot : all clients test_mse on the same axes

Usage:
    uv run python src/data/plot_training_curve.py
    uv run python src/data/plot_training_curve.py --session 20260312150555
    uv run python src/data/plot_training_curve.py --device mps
"""
import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")           # headless (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── project root & shared imports ────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.shared.common import cfg
from src.models.split_lstm import ClientLSTM, ServerHead


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_round(path: str) -> int:
    """Extract zero-padded round from periodic filename: client_1_round_0090.pth → 90"""
    stem = Path(path).stem           # e.g. 'client_1_round_0090'
    parts = stem.split("_")
    try:
        idx = parts.index("round")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return 0


def _load_ckpt(path: str, device: torch.device):
    """Load a checkpoint and return (state_dict, train_loss, round_num)."""
    raw = torch.load(path, map_location=device, weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"], float(raw.get("loss", float("nan"))), raw.get("round", 0)
    # Legacy bare state_dict
    return raw, float("nan"), _parse_round(path)


def _find_session(session_id: str | None) -> Path:
    """Return the session directory to use (bestweights/YYYYMMDD...)."""
    bw = project_root / "bestweights"
    if session_id:
        p = bw / session_id
        if not p.is_dir():
            raise FileNotFoundError(f"Session directory not found: {p}")
        return p

    # Auto-select latest: find all 14-digit directories (YYYYMMDDhhmmss)
    import re
    sessions = sorted([
        d for d in bw.iterdir() 
        if d.is_dir() and re.match(r"^\d{14}$", d.name)
    ])
    
    if not sessions:
        raise FileNotFoundError(f"No valid session directories (YYYYMMDD...) found in {bw}")
    
    latest = sessions[-1]
    print(f"[DEBUG] Picking latest session: {latest.name}")
    return latest


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper (lightweight — mirrors evaluate_client in run_evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_pair(client_state: dict, server_state: dict,
               hidden_size: int, device: torch.device,
               test_days: int, seq_len: int, input_size: int) -> float:
    """
    Run the split-inference model on the test set.
    Returns average MSE over all test samples.
    """
    num_layers = sum(1 for k in client_state if k.startswith("lstm.weight_ih_l"))

    client_model = ClientLSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    ).to(device)
    client_model.load_state_dict(client_state)
    client_model.eval()

    server_model = ServerHead(hidden_size=hidden_size, output_size=1).to(device)
    server_model.load_state_dict(server_state)
    server_model.eval()

    criterion    = nn.MSELoss()
    data_dir     = project_root / cfg.get("data", {}).get("processed_dir", "dataset/processed")
    features_cfg = cfg.get("data", {}).get(
        "feature_cols", ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]
    )
    total_hours = test_days * 24
    total_loss, total_batches = 0.0, 0

    with torch.no_grad():
        for file in sorted(data_dir.glob("NCL_*.parquet")):
            df = pd.read_parquet(file)
            if "Timestamp" in df.columns:
                df.set_index("Timestamp", inplace=True)
                df.index = pd.to_datetime(df.index)
            df["future_3h_rain"] = df["Rain"].shift(-3).rolling(window=3).sum()

            if len(df) < total_hours + seq_len:
                continue

            test_df = df.iloc[-total_hours:].reset_index(drop=True)
            for idx in range(seq_len, len(test_df) - 3):
                if pd.isna(test_df.iloc[idx]["future_3h_rain"]):
                    continue
                target_val = float(test_df.iloc[idx]["future_3h_rain"])
                window     = test_df.iloc[idx - seq_len : idx]
                feat = (
                    window[features_cfg]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0).values
                )
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor([[target_val]], dtype=torch.float32).to(device)

                smashed = client_model(x)
                pred    = server_model(smashed)
                total_loss    += criterion(pred, y).item()
                total_batches += 1

    return total_loss / total_batches if total_batches > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str, default=None,
                        help="Session ID (folder name). Defaults to latest.")
    parser.add_argument("--device",  type=str, default="cpu",
                        help="'cpu' or 'mps'")
    args   = parser.parse_args()
    device = torch.device(args.device)

    # Config
    test_days   = cfg.get("data",  {}).get("test_days",   14)
    seq_len     = cfg.get("model", {}).get("seq_len",     24)
    input_size  = cfg.get("model", {}).get("input_size",   5)
    hidden_size = cfg.get("model", {}).get("hidden_size", 64)

    session_dir  = _find_session(args.session)
    periodic_dir = session_dir / "periodic"
    session_name = session_dir.name

    if not periodic_dir.is_dir():
        print(f"[ERROR] No periodic/ dir found in {session_dir}")
        sys.exit(1)

    print(f"\n📂 Session   : {session_name}")
    print(f"⚙️  Device    : {device}")
    print(f"📅 Test days : {test_days}  |  seq_len={seq_len}\n")

    # ── Collect server periodic checkpoints ──────────────────────────────────
    server_ckpts = {
        _parse_round(p): p
        for p in sorted(glob.glob(str(periodic_dir / "server_round_*.pth")))
    }
    if not server_ckpts:
        print("[ERROR] No server periodic checkpoints found.")
        sys.exit(1)
    server_rounds = sorted(server_ckpts.keys())

    # ── Collect client IDs ───────────────────────────────────────────────────
    client_files = sorted(glob.glob(str(periodic_dir / "client_*_round_*.pth")))
    client_ids: set[int] = set()
    for f in client_files:
        parts = Path(f).stem.split("_")
        try:
            client_ids.add(int(parts[1]))
        except (IndexError, ValueError):
            pass
    client_ids_sorted = sorted(client_ids)

    if not client_ids_sorted:
        print("[ERROR] No client periodic checkpoints found.")
        sys.exit(1)

    print(f"Clients found: {client_ids_sorted}")
    print(f"Server rounds: {server_rounds}\n")

    # ── Evaluate every (client, round) pair ──────────────────────────────────
    # Structure: results[client_id] = [(round, train_loss, test_mse), ...]
    results: dict[int, list[tuple[int, float, float]]] = {}

    for cid in client_ids_sorted:
        ckpts = sorted(
            glob.glob(str(periodic_dir / f"client_{cid}_round_*.pth")),
            key=_parse_round
        )
        if not ckpts:
            print(f"  [SKIP] Client {cid}: no periodic checkpoints")
            continue

        curve = []
        for ckpt_path in ckpts:
            r = _parse_round(ckpt_path)
            # Find closest server round (prefer same, else nearest earlier)
            srv_r = max((sr for sr in server_rounds if sr <= r), default=server_rounds[0])
            srv_path = server_ckpts[srv_r]

            client_state, train_loss, _ = _load_ckpt(ckpt_path, device)
            server_state, _, _          = _load_ckpt(srv_path,  device)

            print(f"  Client {cid} | round {r:04d} | train_loss={train_loss:.4f} "
                  f"| evaluating...", end="", flush=True)
            test_mse = _eval_pair(
                client_state, server_state,
                hidden_size, device, test_days, seq_len, input_size
            )
            print(f"  test_mse={test_mse:.4f}")
            curve.append((r, train_loss, test_mse))

        results[cid] = curve

    if not results:
        print("[ERROR] No results to plot.")
        sys.exit(1)

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_clients = len(results)
    fig_w     = max(10, n_clients * 5)
    fig, axes = plt.subplots(
        1, n_clients + 1,
        figsize=(fig_w + 5, 5),
        squeeze=False
    )
    axes = axes[0]   # flatten

    # Colour palette
    palette = plt.cm.tab10.colors

    # Per-client subplots
    for ax_idx, (cid, curve) in enumerate(sorted(results.items())):
        ax    = axes[ax_idx]
        color = palette[ax_idx % len(palette)]
        rounds     = [c[0] for c in curve]
        train_loss = [c[1] for c in curve]
        test_mse   = [c[2] for c in curve]

        # Train loss line
        ax.plot(rounds, train_loss, "o--", color=color, alpha=0.6,
                linewidth=1.5, markersize=4, label="Train Loss")
        # Test MSE line
        ax.plot(rounds, test_mse, "s-", color=color,
                linewidth=2, markersize=5, label="Test MSE")

        # Mark best round (lowest test MSE)
        valid = [(r, v) for r, v in zip(rounds, test_mse) if not np.isnan(v)]
        if valid:
            best_r, best_v = min(valid, key=lambda x: x[1])
            ax.axvline(best_r, color="red", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.annotate(f"Best\nR={best_r}", xy=(best_r, best_v),
                        xytext=(best_r + 2, best_v * 1.05),
                        fontsize=7, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

        ax.set_title(f"Client {cid}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss / MSE")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#f9f9f9")

    # Combined overlay (last subplot)
    ax_combined = axes[-1]
    for idx, (cid, curve) in enumerate(sorted(results.items())):
        color = palette[idx % len(palette)]
        rounds   = [c[0] for c in curve]
        test_mse = [c[2] for c in curve]
        ax_combined.plot(rounds, test_mse, "s-", color=color,
                         linewidth=2, markersize=4, label=f"Client {cid}")

    ax_combined.set_title("All Clients — Test MSE", fontsize=12, fontweight="bold")
    ax_combined.set_xlabel("Round")
    ax_combined.set_ylabel("Test MSE")
    ax_combined.legend(fontsize=8)
    ax_combined.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_combined.grid(True, alpha=0.3, linestyle="--")
    ax_combined.set_facecolor("#f9f9f9")

    plt.suptitle(
        f"Training Curve ─ Session {session_name}\n"
        f"Dashed = Train Loss  |  Solid = Test MSE  |  Red line = Best Round",
        fontsize=11, y=1.02
    )
    plt.tight_layout()

    out_dir  = project_root / "results" / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"training_curve_{session_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot saved: {out_path}")
    print("\n── Overfitting Summary ──────────────────────────────────────")
    for cid, curve in sorted(results.items()):
        valid = [(r, tl, tm) for r, tl, tm in curve
                 if not np.isnan(tl) and not np.isnan(tm)]
        if not valid:
            continue
        best_r, _, best_tm = min(valid, key=lambda x: x[2])
        last_r, _, last_tm = valid[-1]
        gap = last_tm - best_tm
        flag = "⚠️ Overfitting likely" if gap > 0.005 else "✅ Stable"
        print(f"  Client {cid}: best_round={best_r}  best_testMSE={best_tm:.4f}"
              f"  final_testMSE={last_tm:.4f}  gap={gap:+.4f}  {flag}")
    print()


if __name__ == "__main__":
    main()
