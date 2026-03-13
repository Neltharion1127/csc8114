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
from src.client.data_pipeline import (
    collect_test_indices_capped,
    load_sensor_data,
    partition_client_files,
)
from src.models.split_lstm import ClientLSTM, ServerHead
from src.shared.targets import inverse_target_scalar, is_rain, rain_probability_threshold

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


def find_best_server(session_id: str | None = None) -> str:
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

    if session_id:
        session_dir = bw_dir / session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session not found under bestweights/: {session_id}")
        all_paths = glob.glob(str(session_dir / "server_head_round_*.pth"))
    else:
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


def find_matching_clients(server_path: str, *, allow_cross_session_fallback: bool = False) -> dict[int, str]:
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

    if not client_ids and allow_cross_session_fallback:
        # Fallback: the entire bestweights tree
        print("\u26a0\ufe0f  No client files in server dir. Searching all of bestweights/...")
        fallback_files = (
            glob.glob(str(bw_dir / "best_client_*_round_*_model_*.pth")) +
            glob.glob(str(bw_dir / "*" / "best_client_*_round_*_model_*.pth"))
        )
        client_ids = _collect_ids(fallback_files)
        search_root = None   # signal: use global search per client
    elif not client_ids:
        raise FileNotFoundError(
            f"No client weights found in server session directory: {server_dir}\n"
            "Use --allow-cross-session-fallback to scan all bestweights/ (less strict)."
        )
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
    split_date: pd.Timestamp,
    eval_max_samples: int,
    seq_len: int,
    input_size: int,
    hidden_size: int,
) -> dict:
    """
    Load one client model and evaluate it against all sensor files.
    Returns a dict with mse, mae, accuracy, and sample count.
    """
    criterion = nn.MSELoss()
    prob_threshold = rain_probability_threshold()

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
    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        print(f"  [ERROR] No parquet files found in {data_dir}")
        return {}

    num_clients = max(1, int(cfg.get("federated", {}).get("num_clients", 1)))
    if 1 <= client_id <= num_clients:
        client_files = partition_client_files(
            [str(p) for p in all_files],
            client_id=client_id,
            num_clients=num_clients,
        )
        client_files = [Path(p) for p in client_files]
    else:
        print(
            f"  [WARNING] Client ID {client_id} is outside configured range 1..{num_clients}. "
            "Falling back to full dataset for evaluation."
        )
        client_files = all_files

    print(f"[Client {client_id}] Evaluating {len(client_files)}/{len(all_files)} sensor files")
    if not client_files:
        print(f"  [ERROR] Client {client_id}: assigned 0 sensor files for evaluation.")
        return {}

    active_features = cfg.get("data", {}).get(
        "feature_cols", ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]
    )

    total_loss, total_batches = 0.0, 0
    all_targets, all_preds = [], []
    sensor_data_cache: dict[Path, pd.DataFrame] = {}
    for file in client_files:
        sensor_data_cache[file] = load_sensor_data(str(file))
    all_combined = pd.concat(sensor_data_cache.values())
    feat_mean = all_combined[active_features].mean().values
    feat_std = all_combined[active_features].std().values + 1e-9

    with torch.no_grad():
        for file in client_files:
            sensor_id = file.stem
            df = sensor_data_cache[file]
            test_indices = collect_test_indices_capped(
                df,
                split_date,
                eval_max_samples=eval_max_samples,
                min_history=seq_len,
                horizon=3,
            )
            if len(test_indices) == 0:
                print(f"  [WARNING] No valid test indices in {sensor_id} — skipped")
                continue

            for idx in test_indices:
                target_val = float(df["future_3h_rain"].iloc[int(idx)])
                window_data = df.iloc[int(idx) - seq_len : int(idx)]
                features = (
                    window_data[active_features]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .values
                )
                features = (features - feat_mean) / feat_std
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                y = torch.tensor([[target_val]], dtype=torch.float32).to(device)

                smashed = client_model(x)
                rain_logit, rain_amount = server_model(smashed)
                rain_prob = torch.sigmoid(rain_logit).item()
                pred_val = inverse_target_scalar(rain_amount.item()) if rain_prob >= prob_threshold else 0.0
                pred = torch.tensor([[pred_val]], dtype=torch.float32, device=device)

                total_loss += criterion(pred, y).item()
                total_batches += 1
                all_targets.append(target_val)
                all_preds.append(pred_val)

    if total_batches == 0:
        print(f"  [ERROR] Client {client_id}: No test samples evaluated.")
        return {}

    targets_arr = np.array(all_targets)
    preds_arr   = np.array(all_preds)
    mse      = total_loss / total_batches
    mae      = float(np.mean(np.abs(targets_arr - preds_arr)))
    y_true = np.array([1 if is_rain(t) else 0 for t in targets_arr], dtype=np.int32)
    y_pred = np.array([1 if is_rain(p) else 0 for p in preds_arr], dtype=np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    accuracy = float(((tp + tn) / max(1, len(y_true))) * 100.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    f1 = (
        float(2.0 * recall * precision / (recall + precision))
        if not np.isnan(recall) and not np.isnan(precision) and (recall + precision) > 0
        else float("nan")
    )

    return {
        "client_id":   client_id,
        "best_round":  saved_round,
        "train_loss":  saved_loss,
        "samples":     total_batches,
        "mse":         mse,
        "mae":         mae,
        "accuracy":    accuracy,
        "recall":      recall,
        "precision":   precision,
        "f1":          f1,
        "tp":          tp,
        "fn":          fn,
        "fp":          fp,
        "tn":          tn,
    }


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to run on: 'cpu' or 'mps'"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Evaluate a specific session ID under bestweights/ and results/ (recommended).",
    )
    parser.add_argument(
        "--allow-cross-session-fallback",
        action="store_true",
        help="Allow searching client checkpoints across all sessions if current session lacks clients.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("   🚀 AUTO EVALUATION — ALL CLIENTS (TEST DATA)   ")
    print("=" * 55)

    # Config
    test_days   = cfg.get("data",  {}).get("test_days",   14)
    end_date_str = cfg.get("data_download", {}).get("end_date", "2026-03-10T00:00:00")
    split_date  = pd.Timestamp(end_date_str).tz_localize(None) - pd.Timedelta(days=test_days)
    eval_max_samples = max(0, int(cfg.get("training", {}).get("eval_max_samples_per_sensor", 0)))
    seq_len     = cfg.get("model", {}).get("seq_len",     24)
    input_size  = cfg.get("model", {}).get("input_size",   5)
    hidden_size = cfg.get("model", {}).get("hidden_size", 64)
    head_width  = cfg.get("model", {}).get("server_head_width", 64)
    head_dropout = cfg.get("model", {}).get("server_head_dropout", 0.1)
    device      = torch.device(args.device)
    print(
        f"[INFO] Split Date: {split_date} | seq_len={seq_len} | device={device} | "
        f"per_sensor_cap={eval_max_samples if eval_max_samples > 0 else 'FULL'}"
    )

    # ── Step 1: Auto-select the latest server model ──────────────────
    server_path = find_best_server(session_id=args.session)
    selected_session = Path(server_path).parent.name if Path(server_path).parent != (project_root / "bestweights") else "flat_bestweights"

    # ── Step 2: Find all client models from the same session ─────────
    client_map = find_matching_clients(
        server_path,
        allow_cross_session_fallback=args.allow_cross_session_fallback,
    )   # {client_id: path}
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

    server_model = ServerHead(
        hidden_size=hidden_size,
        output_size=1,
        head_width=head_width,
        dropout=head_dropout,
    ).to(device)
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
            split_date=split_date,
            eval_max_samples=eval_max_samples,
            seq_len=seq_len,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        if result:
            all_results.append(result)
            print(f"  ✅ Client {cid} | Samples={result['samples']:,} | "
                  f"MSE={result['mse']:.4f} | MAE={result['mae']:.4f} mm | "
                  f"Accuracy={result['accuracy']:.2f}% | "
                  f"Recall={result['recall']:.3f} | Precision={result['precision']:.3f} | F1={result['f1']:.3f}")

    # ── Step 5: Print combined summary ─────────────────────────────────
    if not all_results:
        print("\n[ERROR] No clients were successfully evaluated.")
        return

    W = 70  # table width
    print("\n" + "=" * W)
    print("\U0001f3c6  FINAL EVALUATION REPORT (14-DAY TEST SET)")
    print(f"   Session           : {selected_session}")
    print(f"   Server checkpoint : {Path(server_path).name}  (round={server_round})")
    print("=" * W)
    hdr = (f"  {'Client':<8} {'BestRound':>10} {'TrainLoss':>10}"
           f"  {'Samples':>8}  {'MSE':>8}  {'MAE':>8}  {'Accuracy':>10}  {'F1':>8}")
    sep = (f"  {'──────':<8} {'─────────':>10} {'─────────':>10}"
           f"  {'───────':>8}  {'───────':>8}  {'───────':>8}  {'────────':>10}  {'──────':>8}")
    print(hdr)
    print(sep)
    for r in all_results:
        br  = str(r['best_round'])
        tl  = f"{r['train_loss']:.4f}" if not np.isnan(r['train_loss']) else "N/A"
        print(f"  {r['client_id']:<8} {br:>10} {tl:>10}"
              f"  {r['samples']:>8,}  {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['accuracy']:>9.2f}%  {r['f1']:>8.4f}")

    if len(all_results) > 1:
        avg_mse = np.mean([r["mse"]      for r in all_results])
        avg_mae = np.mean([r["mae"]      for r in all_results])
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_f1  = np.mean([r["f1"] for r in all_results if not np.isnan(r["f1"])]) if any(not np.isnan(r["f1"]) for r in all_results) else float("nan")
        tot     = sum(    r["samples"]   for r in all_results)
        print(sep)
        print(f"  {'AVERAGE':<8} {'':>10} {'':>10}"
              f"  {tot:>8,}  {avg_mse:>8.4f}  {avg_mae:>8.4f}  {avg_acc:>9.2f}%  {avg_f1:>8.4f}")
    print("=" * W + "\n")

    results_dir = project_root / "results" / selected_session
    results_dir.mkdir(parents=True, exist_ok=True)

    report_csv = results_dir / f"evaluation_report_{selected_session}.csv"
    pd.DataFrame(all_results).to_csv(report_csv, index=False)
    report_json = results_dir / f"evaluation_report_{selected_session}.json"
    summary = {
        "session": selected_session,
        "server_checkpoint": Path(server_path).name,
        "server_round": server_round,
        "device": str(device),
        "split_date": str(split_date),
        "eval_max_samples_per_sensor": eval_max_samples,
        "num_clients_evaluated": len(all_results),
        "clients": all_results,
    }
    with open(report_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved evaluation CSV : {report_csv}")
    print(f"[INFO] Saved evaluation JSON: {report_json}")


if __name__ == "__main__":
    evaluate()
