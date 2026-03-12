import os
import sys
import glob
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# --- Configuration & Paths ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.shared.common import cfg
from src.models.split_lstm import ClientLSTM, ServerHead

def _parse_timestamp(path: str) -> str:
    """
    Extract the trailing timestamp token from any model filename.
      server_head_round_9_20260312131020.pth       -> '20260312131020'
      best_client_1_round_9_model_20260312131018.pth -> '20260312131018'
    """
    return Path(path).stem.split("_")[-1]


def _parse_round(path: str) -> int:
    """
    Extract the round number from a client or server filename.
      best_client_1_round_9_model_20260312131018.pth  -> 9
      server_head_round_9_20260312131020.pth           -> 9
    Returns 0 if not parseable.
    """
    parts = Path(path).stem.split("_")
    try:
        # 'round' token is always followed by the numeric round number
        idx = parts.index("round")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return 0


def find_best_server() -> str:
    """
    Find the most recent server model in bestweights/.
    Supports both layouts:
      Flat   : bestweights/server_head_round_<N>_<stamp>.pth
      Session: bestweights/<session>/server_head_round_<N>_<stamp>.pth

    Sorting priority: session-subdir mtime (if present) > embedded timestamp.
    Returns the path of the latest server model.
    """
    bw_dir = project_root / "bestweights"
    if not bw_dir.exists():
        raise FileNotFoundError(f"Cannot find bestweights directory at {bw_dir}")

    # Collect from both flat root and one level of subdirectories
    flat_paths   = glob.glob(str(bw_dir / "server_head_round_*.pth"))
    subdir_paths = glob.glob(str(bw_dir / "*" / "server_head_round_*.pth"))
    all_paths    = flat_paths + subdir_paths

    if not all_paths:
        raise FileNotFoundError(
            f"No server weights found in {bw_dir}.\n"
            "Expected: server_head_round_<N>_<stamp>.pth  (flat or inside a session subdir)"
        )

    def _sort_key(p: str):
        parent = Path(p).parent
        folder_mtime = parent.stat().st_mtime if parent != bw_dir else 0.0
        return (folder_mtime, _parse_timestamp(p), _parse_round(p))

    all_paths.sort(key=_sort_key)
    latest_server = all_paths[-1]

    parent = Path(latest_server).parent
    rel    = f"{parent.name}/{Path(latest_server).name}" if parent != bw_dir else Path(latest_server).name
    print(f"\u2705 Latest Server Model    : {rel}")
    return latest_server


def find_matching_clients(server_path: str) -> dict[int, str]:
    """
    Find the best client model for every client ID.
    Searches in the same directory as the server model (flat or session subdir).

    "Best" = highest round number saved (last checkpoint to beat previous best loss).
    If multiple files share the same round, the one with the latest timestamp wins.

    Falls back to searching ALL of bestweights/ if the server's dir has no client files.
    """
    bw_dir      = project_root / "bestweights"
    server_dir  = Path(server_path).parent          # flat root OR session subdir
    is_session  = server_dir != bw_dir
    location    = server_dir.name if is_session else "(flat bestweights/)"
    print(f"\U0001f511 Looking for clients in : {location}")

    def _collect_ids(file_list):
        ids = set()
        for f in file_list:
            parts = Path(f).stem.split("_")
            try:
                ids.add(int(parts[2]))   # index 2 is always the numeric client ID
            except (IndexError, ValueError):
                pass
        return ids

    # Primary search: same directory as server
    primary_files = glob.glob(str(server_dir / "best_client_*_round_*_model_*.pth"))
    client_ids    = _collect_ids(primary_files)

    if not client_ids:
        # Fallback: the entire bestweights tree
        print("\u26a0\ufe0f  No client files in server dir. Searching all of bestweights/...")
        fallback_files = (
            glob.glob(str(bw_dir / "best_client_*_round_*_model_*.pth")) +
            glob.glob(str(bw_dir / "*" / "best_client_*_round_*_model_*.pth"))
        )
        client_ids = _collect_ids(fallback_files)
        search_root = None   # signal: use global search per client
    else:
        search_root = server_dir

    if not client_ids:
        raise FileNotFoundError(
            "No client weight files found anywhere in bestweights/.\n"
            "Expected: best_client_<ID>_round_<N>_model_<stamp>.pth"
        )

    matched: dict[int, str] = {}

    for cid in sorted(client_ids):
        if search_root is not None:
            candidates = glob.glob(str(search_root / f"best_client_{cid}_round_*_model_*.pth"))
        else:
            candidates = (
                glob.glob(str(bw_dir / f"best_client_{cid}_round_*_model_*.pth")) +
                glob.glob(str(bw_dir / "*" / f"best_client_{cid}_round_*_model_*.pth"))
            )

        if not candidates:
            print(f"\u26a0\ufe0f  Client {cid}: No model files found \u2014 skipping")
            continue

        # Pick the file with the highest round number;
        # break ties by latest timestamp (lexicographic on the stamp token)
        candidates.sort(key=lambda p: (_parse_round(p), _parse_timestamp(p)))
        chosen = candidates[-1]
        tag    = f"round {_parse_round(chosen)}, ts={_parse_timestamp(chosen)}"
        print(f"\U0001f464 Client {cid} Best Model    : {Path(chosen).name}  ({tag})")
        matched[cid] = chosen

    if not matched:
        raise FileNotFoundError("Could not find any client weight files.")

    return matched


def evaluate_client(
    client_id: int,
    client_path: str,
    server_model: nn.Module,
    device: torch.device,
    test_days: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
) -> dict:
    """
    Load one client model and evaluate it against all sensor files.
    Returns a dict with mse, mae, accuracy, and sample count.
    """
    criterion = nn.MSELoss()

    # ── Load checkpoint (supports both new dict format and old bare state_dict) ──
    raw = torch.load(client_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        # New format: full checkpoint dict
        state_dict   = raw["model_state_dict"]
        saved_round  = raw.get("round", "?")            # best round recorded
        saved_loss   = raw.get("loss",  float("nan"))  # best train loss recorded
        ckpt_cfg     = raw.get("config", {})
        hidden_size  = ckpt_cfg.get("hidden_size", hidden_size)   # override if saved
        print(f"\n[Client {client_id}] \U0001f4c4 Checkpoint dict \u2014 best_round={saved_round}, train_loss={saved_loss:.4f}")
    else:
        # Old format: bare state_dict
        state_dict   = raw
        saved_round  = "N/A"
        saved_loss   = float("nan")
        print(f"\n[Client {client_id}] \U0001f4c4 Legacy checkpoint (bare state_dict)")

    num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
    client_model = ClientLSTM(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
    ).to(device)
    client_model.load_state_dict(state_dict)
    client_model.eval()
    print(f"[Client {client_id}] Architecture: hidden={hidden_size}, layers={num_layers}")

    data_dir = project_root / cfg.get("data", {}).get("processed_dir", "dataset/processed")
    client_files = sorted(data_dir.glob("NCL_*.parquet"))
    active_features = cfg.get("data", {}).get(
        "feature_cols", ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]
    )

    total_loss, total_batches = 0.0, 0
    all_targets, all_preds = [], []
    total_hours = test_days * 24

    with torch.no_grad():
        for file in client_files:
            sensor_id = file.stem
            df = pd.read_parquet(file)

            if "Timestamp" in df.columns:
                df.set_index("Timestamp", inplace=True)
                df.index = pd.to_datetime(df.index)
            df["future_3h_rain"] = df["Rain"].shift(-3).rolling(window=3).sum()

            if len(df) < total_hours + seq_len:
                print(f"  [WARNING] Not enough data in {sensor_id} — skipped")
                continue

            test_df = df.iloc[-total_hours:].reset_index(drop=True)

            for idx in range(seq_len, len(test_df) - 3):
                if pd.isna(test_df.iloc[idx]["future_3h_rain"]):
                    continue
                target_val = float(test_df.iloc[idx]["future_3h_rain"])
                window_data = test_df.iloc[idx - seq_len : idx]
                features = (
                    window_data[active_features]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .values
                )
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor([[target_val]], dtype=torch.float32).to(device)

                smashed = client_model(x)
                pred    = server_model(smashed)

                total_loss += criterion(pred, y).item()
                total_batches += 1
                all_targets.append(target_val)
                all_preds.append(pred.item())

    if total_batches == 0:
        print(f"  [ERROR] Client {client_id}: No test samples evaluated.")
        return {}

    targets_arr = np.array(all_targets)
    preds_arr   = np.array(all_preds)
    mse      = total_loss / total_batches
    mae      = float(np.mean(np.abs(targets_arr - preds_arr)))
    accuracy = float(np.mean((targets_arr > 0.1) == (preds_arr > 0.1)) * 100)

    return {
        "client_id":   client_id,
        "best_round":  saved_round,
        "train_loss":  saved_loss,
        "samples":     total_batches,
        "mse":         mse,
        "mae":         mae,
        "accuracy":    accuracy,
    }


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on: 'cpu' or 'mps'"
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("   🚀 AUTO EVALUATION — ALL CLIENTS (TEST DATA)   ")
    print("=" * 55)

    # Config
    test_days   = cfg.get("data",  {}).get("test_days",   14)
    seq_len     = cfg.get("model", {}).get("seq_len",     24)
    input_size  = cfg.get("model", {}).get("input_size",   5)
    hidden_size = cfg.get("model", {}).get("hidden_size", 64)
    device      = torch.device(args.device)
    print(f"[INFO] Test Window : last {test_days} days  |  seq_len={seq_len}  |  device={device}")

    # ── Step 1: Auto-select the latest server model ──────────────────
    server_path = find_best_server()

    # ── Step 2: Find all client models from the same session ─────────
    client_map = find_matching_clients(server_path)   # {client_id: path}
    print(f"\n[INFO] Found {len(client_map)} client model(s) to evaluate: {sorted(client_map.keys())}")

    # ── Step 3: Load server model once (shared across all clients) ───
    server_raw = torch.load(server_path, map_location=device, weights_only=True)
    if isinstance(server_raw, dict) and "model_state_dict" in server_raw:
        server_state  = server_raw["model_state_dict"]
        server_round  = server_raw.get("round", "?")
        print(f"[INFO] Server checkpoint dict \u2014 round={server_round}")
    else:
        server_state  = server_raw
        server_round  = "N/A"
        print("[INFO] Server: legacy checkpoint (bare state_dict)")

    server_model = ServerHead(hidden_size=hidden_size, output_size=1).to(device)
    server_model.load_state_dict(server_state)
    server_model.eval()
    print(f"[INFO] Server model loaded and set to eval mode.")

    # ── Step 4: Evaluate each client ─────────────────────────────────
    all_results = []
    for cid, cpath in sorted(client_map.items()):
        print(f"\n{'─'*55}")
        print(f"  Evaluating Client {cid} ...")
        print(f"{'─'*55}")
        result = evaluate_client(
            client_id=cid,
            client_path=cpath,
            server_model=server_model,
            device=device,
            test_days=test_days,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        if result:
            all_results.append(result)
            print(f"  ✅ Client {cid} | Samples={result['samples']:,} | "
                  f"MSE={result['mse']:.4f} | MAE={result['mae']:.4f} mm | "
                  f"Accuracy={result['accuracy']:.2f}%")

    # ── Step 5: Print combined summary ─────────────────────────────────
    if not all_results:
        print("\n[ERROR] No clients were successfully evaluated.")
        return

    W = 70  # table width
    print("\n" + "=" * W)
    print("\U0001f3c6  FINAL EVALUATION REPORT (14-DAY TEST SET)")
    print(f"   Server checkpoint : {Path(server_path).name}  (round={server_round})")
    print("=" * W)
    hdr = (f"  {'Client':<8} {'BestRound':>10} {'TrainLoss':>10}"
           f"  {'Samples':>8}  {'MSE':>8}  {'MAE':>8}  {'Accuracy':>10}")
    sep = (f"  {'──────':<8} {'─────────':>10} {'─────────':>10}"
           f"  {'───────':>8}  {'───────':>8}  {'───────':>8}  {'────────':>10}")
    print(hdr)
    print(sep)
    for r in all_results:
        br  = str(r['best_round'])
        tl  = f"{r['train_loss']:.4f}" if not np.isnan(r['train_loss']) else "N/A"
        print(f"  {r['client_id']:<8} {br:>10} {tl:>10}"
              f"  {r['samples']:>8,}  {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['accuracy']:>9.2f}%")

    if len(all_results) > 1:
        avg_mse = np.mean([r["mse"]      for r in all_results])
        avg_mae = np.mean([r["mae"]      for r in all_results])
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        tot     = sum(    r["samples"]   for r in all_results)
        print(sep)
        print(f"  {'AVERAGE':<8} {'':>10} {'':>10}"
              f"  {tot:>8,}  {avg_mse:>8.4f}  {avg_mae:>8.4f}  {avg_acc:>9.2f}%")
    print("=" * W + "\n")


if __name__ == "__main__":
    evaluate()
