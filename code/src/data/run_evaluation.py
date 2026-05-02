import sys
import glob
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

# --- Configuration & Paths ---
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.shared.common import cfg, feature_cols_from_cfg, get_nested
from src.client.data_pipeline import (
    FUTURE_RAIN_COL,
    collect_eval_indices_capped,
    collect_test_indices_capped,
    get_dataset_split,
    load_sensor_data,
    partition_client_files,
)
from src.models.split_lstm import ClientLSTM, ServerHead
from src.shared.targets import (
    inverse_target_scalar,
    is_rain,
    rain_probability_threshold,
    rain_threshold_mm,
    target_transform_mode,
)


def _normalize_report_tag(tag: str) -> str:
    raw = str(tag).strip()
    if not raw:
        return ""
    safe = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw)
    return safe.strip("_")


def _parse_threshold_list(spec: str) -> list[float]:
    raw = str(spec).strip()
    if not raw:
        return []
    if ":" in raw:
        parts = [p.strip() for p in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("--scan-thresholds with ':' must be start:end:step")
        start, end, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise ValueError("--scan-thresholds step must be > 0")
        values: list[float] = []
        x = start
        # epsilon for float endpoint inclusion
        while x <= end + 1e-12:
            values.append(float(round(x, 6)))
            x += step
        return [v for v in values if 0.0 <= v <= 1.0]

    values = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        v = float(t)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"threshold out of range [0,1]: {v}")
        values.append(float(round(v, 6)))
    return sorted(set(values))


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


def _find_latest_session_id() -> str:
    bw_dir = project_root / "bestweights"
    sessions = sorted([d for d in bw_dir.glob("20*") if d.is_dir()], key=lambda p: p.name)
    if not sessions:
        raise FileNotFoundError(f"No session folders found under {bw_dir}")
    return sessions[-1].name


def find_periodic_pair(
    *,
    session_id: str,
    num_clients: int | None = None,
    target_round: int | None = None,
    scenario_id: str | None = None,
) -> tuple[int, str, dict[int, str]] | None:
    """
    Find a strictly paired periodic checkpoint set:
      - server_round_<R>.pth
      - client_<cid>_round_<R>.pth for all available clients at round R,
        or all cid in [1..num_clients] if num_clients is provided.

    If scenario_id is given, only the matching scenario subdirectory is searched.
    """
    session_dir = project_root / "bestweights" / session_id
    if not session_dir.is_dir():
        return None

    periodic_roots: list[Path] = []
    if scenario_id:
        # Narrow search to the specific scenario subdirectory only.
        scenario_periodic = session_dir / scenario_id / "periodic"
        if scenario_periodic.is_dir():
            periodic_roots.append(scenario_periodic)
    else:
        direct = session_dir / "periodic"
        if direct.is_dir():
            periodic_roots.append(direct)
        for sub in sorted(session_dir.glob("*/periodic")):
            if sub.is_dir():
                periodic_roots.append(sub)
    if not periodic_roots:
        return None

    best_candidate: tuple[int, float, str, dict[int, str]] | None = None
    # Sort newest-first so ties prefer most recent scenario folder.
    periodic_roots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for periodic_dir in periodic_roots:
        server_by_round: dict[int, str] = {}
        for p in periodic_dir.glob("server_round_*.pth"):
            try:
                r = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            server_by_round[r] = str(p)
        if not server_by_round:
            continue

        client_by_round: dict[int, dict[int, str]] = {}
        for p in periodic_dir.glob("client_*_round_*.pth"):
            parts = p.stem.split("_")
            try:
                cid = int(parts[1])
                rnd = int(parts[3])
            except (IndexError, ValueError):
                continue
            client_by_round.setdefault(cid, {})[rnd] = str(p)

        if target_round is not None:
            candidate_rounds = [target_round]
        else:
            candidate_rounds = sorted(server_by_round.keys(), reverse=True)

        for rnd in candidate_rounds:
            server_path = server_by_round.get(rnd)
            if not server_path:
                continue
            cmap: dict[int, str] = {}
            if num_clients is None:
                expected_cids = sorted(cid for cid, by_round in client_by_round.items() if rnd in by_round)
            else:
                expected_cids = list(range(1, num_clients + 1))
            if not expected_cids:
                continue
            ok = True
            for cid in expected_cids:
                cpath = client_by_round.get(cid, {}).get(rnd)
                if not cpath:
                    ok = False
                    break
                cmap[cid] = cpath
            if not ok:
                continue

            candidate = (rnd, periodic_dir.stat().st_mtime, server_path, cmap)
            if best_candidate is None or candidate[:2] > best_candidate[:2]:
                best_candidate = candidate
            break

    if best_candidate is None:
        return None
    return best_candidate[0], best_candidate[2], best_candidate[3]


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
        
        # Support both flat layout and nested scenario layout (e.g. session/01_seed42/server_head...)
        all_paths = glob.glob(str(session_dir / "server_head_round_*.pth"))
        if not all_paths:
            all_paths = glob.glob(str(session_dir / "**/server_head_round_*.pth"), recursive=True)
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

    # Primary search: same directory as server or subdirectories
    primary_files = glob.glob(str(server_dir / "best_client_*_round_*_model_*.pth"))
    if not primary_files:
         primary_files = glob.glob(str(server_dir / "**/best_client_*_round_*_model_*.pth"), recursive=True)
         
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


def _metrics_from_cm(tp: int, fn: int, fp: int, tn: int) -> dict[str, float | int]:
    total = max(1, tp + fn + fp + tn)
    accuracy = float((tp + tn) / total * 100.0)
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = (
        float(2.0 * recall * precision / (recall + precision))
        if (recall + precision) > 0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }


def _class_metrics_at_threshold(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (probs >= float(threshold)).astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    metrics = _metrics_from_cm(tp, fn, fp, tn)
    metrics["pred_positive_rate"] = float(y_pred.mean())
    metrics["threshold"] = float(threshold)
    return metrics


def _resolve_eval_settings(server_raw: dict | object) -> tuple[dict, str]:
    snapshot = (
        server_raw.get("config_snapshot")
        if isinstance(server_raw, dict) and isinstance(server_raw.get("config_snapshot"), dict)
        else None
    )
    source = "checkpoint_snapshot" if snapshot is not None else "runtime_config"

    def _setting(path: tuple[str, ...], default):
        if snapshot is not None:
            snap_value = get_nested(snapshot, path, None)
            if snap_value is not None:
                return snap_value
        return get_nested(cfg, path, default)

    settings = {
        "end_date_str": str(_setting(("data_download", "end_date"), "2026-03-10T00:00:00")),
        "eval_max_samples": max(0, int(_setting(("training", "eval_max_samples_per_sensor"), 0))),
        "seq_len": int(_setting(("model", "seq_len"), 24)),
        "horizon": max(1, int(_setting(("model", "horizon"), 3))),
        "input_size": int(_setting(("model", "input_size"), 5)),
        "lstm_dropout": float(_setting(("model", "lstm_dropout"), _setting(("model", "dropout"), 0.3))),
        "hidden_size": int(_setting(("model", "hidden_size"), 64)),
        "head_width": int(_setting(("model", "server_head_width"), 64)),
        "head_dropout": float(_setting(("model", "server_head_dropout"), 0.1)),
        "num_clients": max(1, int(_setting(("federated", "num_clients"), 1))),
        "processed_dir": str(_setting(("data", "processed_dir"), "dataset/processed")),
        "active_features": feature_cols_from_cfg(snapshot),
        "prob_threshold": float(
            _setting(("training", "rain_probability_threshold"), rain_probability_threshold())
        ),
        "rain_threshold": float(
            _setting(("training", "rain_threshold_mm"), rain_threshold_mm())
        ),
        "target_mode": str(
            _setting(("training", "target_transform"), target_transform_mode())
        ).strip().lower(),
        "config_snapshot_used": snapshot is not None,
    }
    server_cfg = server_raw.get("config", {}) if isinstance(server_raw, dict) else {}
    if isinstance(server_cfg, dict) and server_cfg.get("hidden_size") is not None:
        settings["hidden_size"] = int(server_cfg.get("hidden_size"))

    val_end_str = str(_setting(("data", "val_end"), "2025-07-01"))
    split_date = pd.Timestamp(val_end_str)
    settings["split_date"] = split_date
    return settings, source


def evaluate_client(
    client_id: int,
    client_path: str,
    server_model: nn.Module,
    device: torch.device,
    split_date: pd.Timestamp,
    eval_max_samples: int,
    seq_len: int,
    horizon: int,
    input_size: int,
    lstm_dropout: float,
    hidden_size: int,
    num_clients: int,
    processed_dir: str,
    active_features: list[str],
    prob_threshold: float,
    force_prob_threshold: float | None,
    prefer_checkpoint_threshold: bool,
    eval_phase: str,
    scan_thresholds: list[float] | None,
    rain_threshold: float,
    target_mode: str,
) -> dict:
    """
    Load one client model and evaluate it against all sensor files.
    Returns a dict with mse, mae, accuracy, and sample count.
    """
    criterion = nn.MSELoss()

    # ── Load checkpoint (supports both new dict format and old bare state_dict) ──
    raw = torch.load(client_path, map_location="cpu", weights_only=True)
    client_prob_threshold: float | None = None
    if isinstance(raw, dict) and "model_state_dict" in raw:
        # New format: full checkpoint dict
        state_dict   = raw["model_state_dict"]
        saved_round  = raw.get("round", "?")            # best round recorded
        saved_loss   = raw.get("loss",  float("nan"))  # best train loss recorded
        ckpt_cfg     = raw.get("config", {})
        hidden_size  = ckpt_cfg.get("hidden_size", hidden_size)   # override if saved
        cls_metrics = raw.get("classification_metrics", {})
        if isinstance(cls_metrics, dict) and cls_metrics.get("threshold") is not None:
            try:
                client_prob_threshold = float(cls_metrics.get("threshold"))
            except (TypeError, ValueError):
                client_prob_threshold = None
        print(f"\n[Client {client_id}] \U0001f4c4 Checkpoint dict \u2014 best_round={saved_round}, train_loss={saved_loss:.4f}")
    else:
        # Old format: bare state_dict
        state_dict   = raw
        saved_round  = "N/A"
        saved_loss   = float("nan")
        print(f"\n[Client {client_id}] \U0001f4c4 Legacy checkpoint (bare state_dict)")
    if force_prob_threshold is not None:
        effective_prob_threshold = float(force_prob_threshold)
        prob_threshold_source = "forced_cli"
    elif prefer_checkpoint_threshold and client_prob_threshold is not None:
        effective_prob_threshold = client_prob_threshold
        prob_threshold_source = "client_checkpoint"
    else:
        effective_prob_threshold = prob_threshold
        prob_threshold_source = "config"

    num_layers = sum(1 for k in state_dict if k.startswith("lstm.weight_ih_l"))
    client_model = ClientLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lstm_dropout=lstm_dropout,
    ).to(device)
    client_model.load_state_dict(state_dict)
    client_model.eval()
    print(f"[Client {client_id}] Architecture: hidden={hidden_size}, layers={num_layers}")

    data_dir = project_root / processed_dir
    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        print(f"  [ERROR] No parquet files found in {data_dir}")
        return {}

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

    total_loss, total_batches = 0.0, 0
    all_targets, all_preds, all_probs = [], [], []
    
    # Initialize buckets for 12 months
    month_stats = {m: {"loss": 0.0, "batches": 0, "targets": [], "probs": []} for m in range(1, 13)}
    
    sensor_data_cache: dict[Path, pd.DataFrame] = {}
    for file in client_files:
        sensor_data_cache[file] = load_sensor_data(str(file), horizon=horizon)
    # Use TRAIN-period rows only — matches the normalization used during training
    train_frames = [
        df[np.array([get_dataset_split(ts) == "TRAIN" for ts in df.index])]
        for df in sensor_data_cache.values()
    ]
    train_combined = pd.concat([f for f in train_frames if not f.empty])
    feat_mean = train_combined[active_features].mean().values
    feat_std = train_combined[active_features].std().values + 1e-9

    with torch.no_grad():
        for file in client_files:
            sensor_id = file.stem
            df = sensor_data_cache[file]
            if eval_phase == "VAL":
                eval_indices = collect_eval_indices_capped(
                    df,
                    target_phase="VAL",
                    eval_max_samples=eval_max_samples,
                    min_history=seq_len,
                    horizon=horizon,
                )
            else:
                eval_indices = collect_test_indices_capped(
                    df,
                    eval_max_samples=eval_max_samples,
                    min_history=seq_len,
                    horizon=horizon,
                )
            if len(eval_indices) == 0:
                print(f"  [WARNING] No valid {eval_phase} indices in {sensor_id} — skipped")
                continue
            
            for idx in eval_indices:
                # Resolve month from current timestamp
                # Note: idx is the index label in indices, we need the timestamp at that index
                try:
                    ts = df.index[int(idx)]
                    m_idx = ts.month
                except:
                    m_idx = 1 # Fallback
                
                target_val = float(df[FUTURE_RAIN_COL].iloc[int(idx)])
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
                raw_pred_val = inverse_target_scalar(rain_amount.item(), mode=target_mode)
                pred_val = raw_pred_val if rain_prob >= effective_prob_threshold else 0.0
                
                loss = criterion(torch.tensor([[pred_val]], device=device), y).item()
                
                # Monthly collectors
                month_stats[m_idx]["loss"] += loss
                month_stats[m_idx]["batches"] += 1
                month_stats[m_idx]["targets"].append(target_val)
                month_stats[m_idx]["probs"].append(rain_prob)

                # Global collectors
                all_targets.append(target_val)
                all_probs.append(rain_prob)
                all_preds.append(pred_val)
                total_loss += loss
                total_batches += 1
            
    # Calculate Final Monthly Details
    monthly_details = []
    
    for m in range(1, 13):
        m_data = month_stats[m]
        if m_data["batches"] > 0:
            m_mse = m_data["loss"] / m_data["batches"]
            m_y_true = [1 if is_rain(t, threshold=rain_threshold) else 0 for t in m_data["targets"]]
            m_y_pred = [1 if p >= effective_prob_threshold else 0 for p in m_data["probs"]]
            
            m_acc = accuracy_score(m_y_true, m_y_pred)
            m_f1  = f1_score(m_y_true, m_y_pred, zero_division=0)
            
            monthly_details.append({
                "Month": f"{m:02d}",
                "MSE": round(m_mse, 6),
                "Acc": round(m_acc, 4),
                "F1": round(m_f1, 4),
                "Samples": m_data["batches"]
            })

    if total_batches == 0:
        print(f"  [ERROR] Client {client_id}: No test samples evaluated.")
        return {}

    targets_arr = np.array(all_targets)
    preds_arr   = np.array(all_preds)
    mse      = total_loss / total_batches
    mae      = float(np.mean(np.abs(targets_arr - preds_arr)))
    rain_mask = np.array([is_rain(t, threshold=rain_threshold) for t in targets_arr])
    if rain_mask.any():
        rain_mse = float(np.mean((targets_arr[rain_mask] - preds_arr[rain_mask]) ** 2))
        rain_mae = float(np.mean(np.abs(targets_arr[rain_mask] - preds_arr[rain_mask])))
    else:
        rain_mse = float("nan")
        rain_mae = float("nan")
    probs_arr = np.array(all_probs)
    y_true = np.array([1 if is_rain(t, threshold=rain_threshold) else 0 for t in targets_arr], dtype=np.int32)
    positive_rate = float(y_true.mean())
    prob_mean = float(probs_arr.mean())
    prob_std = float(probs_arr.std())
    brier = float(np.mean((probs_arr - y_true) ** 2))
    if np.unique(y_true).size >= 2:
        roc_auc = float(roc_auc_score(y_true, probs_arr))
    else:
        roc_auc = 0.5
    if int(y_true.sum()) > 0:
        auprc = float(average_precision_score(y_true, probs_arr))
    else:
        auprc = 0.0

    cls_metrics = _class_metrics_at_threshold(
        y_true=y_true,
        probs=probs_arr,
        threshold=effective_prob_threshold,
    )
    pred_positive_rate = float(cls_metrics["pred_positive_rate"])
    cls_source = "offline_recomputed"
    selected_cls = cls_metrics

    y_pred_op = np.array([1 if is_rain(p, threshold=rain_threshold) else 0 for p in preds_arr], dtype=np.int32)
    op_tp = int(((y_true == 1) & (y_pred_op == 1)).sum())
    op_fn = int(((y_true == 1) & (y_pred_op == 0)).sum())
    op_fp = int(((y_true == 0) & (y_pred_op == 1)).sum())
    op_tn = int(((y_true == 0) & (y_pred_op == 0)).sum())
    op_metrics = _metrics_from_cm(op_tp, op_fn, op_fp, op_tn)

    threshold_scan: list[dict[str, float | int]] = []
    if scan_thresholds:
        for thr in scan_thresholds:
            threshold_scan.append(
                _class_metrics_at_threshold(
                    y_true=y_true,
                    probs=probs_arr,
                    threshold=float(thr),
                )
            )

    return {
        "client_id":   client_id,
        "best_round":  saved_round,
        "train_loss":  saved_loss,
        "samples":     total_batches,
        "mse":         mse,
        "mae":         mae,
        "rain_mse":    rain_mse,
        "rain_mae":    rain_mae,
        "auprc":       auprc,
        "roc_auc":     roc_auc,
        "brier":       brier,
        "positive_rate": positive_rate,
        "pred_positive_rate": pred_positive_rate,
        "prob_mean":   prob_mean,
        "prob_std":    prob_std,
        "accuracy":    selected_cls["accuracy"],
        "recall":      selected_cls["recall"],
        "precision":   selected_cls["precision"],
        "f1":          selected_cls["f1"],
        "tp":          selected_cls["tp"],
        "fn":          selected_cls["fn"],
        "fp":          selected_cls["fp"],
        "tn":          selected_cls["tn"],
        "cls_metric_source": cls_source,
        "offline_accuracy": cls_metrics["accuracy"],
        "offline_recall": cls_metrics["recall"],
        "offline_precision": cls_metrics["precision"],
        "offline_f1": cls_metrics["f1"],
        "offline_tp": cls_metrics["tp"],
        "offline_fn": cls_metrics["fn"],
        "offline_fp": cls_metrics["fp"],
        "offline_tn": cls_metrics["tn"],
        "op_accuracy": op_metrics["accuracy"],
        "op_recall":   op_metrics["recall"],
        "op_precision": op_metrics["precision"],
        "op_f1":       op_metrics["f1"],
        "op_tp":       op_metrics["tp"],
        "op_fn":       op_metrics["fn"],
        "op_fp":       op_metrics["fp"],
        "op_tn":       op_metrics["tn"],
        "prob_threshold": effective_prob_threshold,
        "prob_threshold_source": prob_threshold_source,
        "rain_threshold_mm": rain_threshold,
        "target_transform": target_mode,
        "monthly_details": monthly_details,
        "threshold_scan": threshold_scan,
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
        "--scenario",
        type=str,
        default=None,
        help="Restrict checkpoint search to a specific scenario subdirectory (e.g. '07'). "
             "Required when multiple scenarios share the same session (matrix runs).",
    )
    parser.add_argument(
        "--eval-phase",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Evaluation split to use: test or val (threshold tuning should use val).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Evaluate a specific round. Requires a strict paired periodic checkpoint set.",
    )
    parser.add_argument(
        "--allow-cross-session-fallback",
        action="store_true",
        help="Allow searching client checkpoints across all sessions if current session lacks clients.",
    )
    parser.add_argument(
        "--allow-latest-best-fallback",
        action="store_true",
        help="Allow non-strict latest/best server-client pairing (debug only; less reliable).",
    )
    parser.add_argument(
        "--force-prob-threshold",
        type=float,
        default=None,
        help="Force one shared probability threshold in [0,1] for all clients (ignores checkpoint threshold).",
    )
    parser.add_argument(
        "--prefer-checkpoint-threshold",
        action="store_true",
        help="Prefer per-client checkpoint threshold when available (default uses config threshold).",
    )
    parser.add_argument(
        "--scan-thresholds",
        type=str,
        default="",
        help="Threshold grid for scan (e.g. '0.1,0.2,0.3' or '0.1:0.9:0.05').",
    )
    parser.add_argument(
        "--report-tag",
        type=str,
        default="",
        help="Optional suffix for output report files (e.g. fixedthr034).",
    )
    args = parser.parse_args()
    forced_prob_threshold = None
    if args.force_prob_threshold is not None:
        forced_prob_threshold = float(args.force_prob_threshold)
        if not (0.0 <= forced_prob_threshold <= 1.0):
            raise ValueError("--force-prob-threshold must be in [0, 1].")
    report_tag = _normalize_report_tag(args.report_tag)
    if args.report_tag and not report_tag:
        raise ValueError("Invalid --report-tag: must contain at least one alphanumeric character.")
    eval_phase = str(args.eval_phase).strip().upper()
    scan_thresholds = _parse_threshold_list(args.scan_thresholds)

    print("\n" + "=" * 55)
    print(f"   🚀 AUTO EVALUATION — ALL CLIENTS ({eval_phase} DATA)   ")
    print("=" * 55)

    device = torch.device(args.device)

    # ── Step 1: Select session and checkpoint pairing mode ────────────
    selected_session = args.session or _find_latest_session_id()
    pairing_mode = "strict_periodic"

    num_clients_hint = max(1, int(get_nested(cfg, ("federated", "num_clients"), 1)))
    paired = find_periodic_pair(
        session_id=selected_session,
        num_clients=num_clients_hint,
        target_round=args.round,
        scenario_id=args.scenario or None,
    )
    if paired is not None:
        paired_round, server_path, client_map = paired
        pairing_mode = f"periodic_round_{paired_round}"
        print(f"[INFO] Using strict paired periodic checkpoints at round {paired_round}")
    else:
        round_hint = f", round={args.round}" if args.round is not None else ""
        if not args.allow_latest_best_fallback:
            raise FileNotFoundError(
                f"No strict paired periodic checkpoints found for session={selected_session}{round_hint}. "
                "Use --allow-latest-best-fallback for debug-only non-strict pairing."
            )
        if args.round is not None:
            raise FileNotFoundError(
                f"No strict paired periodic checkpoints found for session={selected_session}, round={args.round}."
            )
        server_path = find_best_server(session_id=selected_session)
        client_map = find_matching_clients(
            server_path,
            allow_cross_session_fallback=args.allow_cross_session_fallback,
        )
        pairing_mode = "latest_best"
        print("[WARN] Falling back to latest/best checkpoint selection (non-strict pairing).")

    # ── Step 2: Summary of selected client checkpoints ───────────────
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

    eval_settings, eval_cfg_source = _resolve_eval_settings(server_raw)
    split_date = eval_settings["split_date"]
    seq_len = eval_settings["seq_len"]
    horizon = eval_settings["horizon"]
    input_size = eval_settings["input_size"]
    lstm_dropout = eval_settings["lstm_dropout"]
    hidden_size = eval_settings["hidden_size"]
    head_width = eval_settings["head_width"]
    head_dropout = eval_settings["head_dropout"]
    eval_max_samples = eval_settings["eval_max_samples"]
    num_clients = eval_settings["num_clients"]
    processed_dir = eval_settings["processed_dir"]
    active_features = eval_settings["active_features"]
    prob_threshold = eval_settings["prob_threshold"]
    rain_threshold = eval_settings["rain_threshold"]
    target_mode = eval_settings["target_mode"]
    if pairing_mode.startswith("periodic_round_"):
        expected_clients = list(range(1, num_clients + 1))
        if sorted(client_map.keys()) != expected_clients:
            raise FileNotFoundError(
                f"Strict periodic pairing requires all clients {expected_clients}, "
                f"but found {sorted(client_map.keys())} in {pairing_mode}."
            )
    threshold_text = (
        f"p(rain)>={forced_prob_threshold:.2f} (forced)"
        if forced_prob_threshold is not None
        else f"p(rain)>={prob_threshold:.2f}"
    )
    print(
        f"[INFO] Eval config source: {eval_cfg_source} | split_date={split_date} | "
        f"seq_len={seq_len} horizon={horizon} | per_sensor_cap={eval_max_samples if eval_max_samples > 0 else 'FULL'} | "
        f"{threshold_text} | rain_mm>{rain_threshold:.2f} | target_transform={target_mode} | phase={eval_phase} | "
        f"threshold_source={'checkpoint_preferred' if args.prefer_checkpoint_threshold else 'config'} | device={device}"
    )
    if forced_prob_threshold is not None:
        print(
            f"[INFO] Forced probability threshold active: p(rain)>={forced_prob_threshold:.2f} "
            "(checkpoint thresholds ignored)"
        )

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
            horizon=horizon,
            input_size=input_size,
            lstm_dropout=lstm_dropout,
            hidden_size=hidden_size,
            num_clients=num_clients,
            processed_dir=processed_dir,
            active_features=active_features,
            prob_threshold=prob_threshold,
            force_prob_threshold=forced_prob_threshold,
            prefer_checkpoint_threshold=args.prefer_checkpoint_threshold,
            eval_phase=eval_phase,
            scan_thresholds=scan_thresholds,
            rain_threshold=rain_threshold,
            target_mode=target_mode,
        )
        if result:
            all_results.append(result)
            print(f"  ✅ Client {cid} | Samples={result['samples']:,} | "
                  f"MSE={result['mse']:.4f} | MAE={result['mae']:.4f} mm | "
                  f"ClsF1={result['f1']:.3f} | PR-AUC={result['auprc']:.3f} | "
                  f"ROC-AUC={result['roc_auc']:.3f} | Brier={result['brier']:.4f}")

    # ── Step 5: Print combined summary ─────────────────────────────────
    if not all_results:
        print("\n[ERROR] No clients were successfully evaluated.")
        return

    threshold_scan_summary: list[dict[str, float | int]] = []
    recommended_threshold: float | None = None
    if scan_thresholds:
        for thr in scan_thresholds:
            tp = fn = fp = tn = 0
            total = 0
            for r in all_results:
                total += int(r["samples"])
                scan_rows = r.get("threshold_scan", [])
                found = next((m for m in scan_rows if abs(float(m["threshold"]) - float(thr)) <= 1e-12), None)
                if found is None:
                    continue
                tp += int(found["tp"])
                fn += int(found["fn"])
                fp += int(found["fp"])
                tn += int(found["tn"])
            metrics = _metrics_from_cm(tp, fn, fp, tn)
            threshold_scan_summary.append(
                {
                    "threshold": float(thr),
                    "samples": int(total),
                    "accuracy": float(metrics["accuracy"]),
                    "recall": float(metrics["recall"]),
                    "precision": float(metrics["precision"]),
                    "f1": float(metrics["f1"]),
                    "tp": int(tp),
                    "fn": int(fn),
                    "fp": int(fp),
                    "tn": int(tn),
                    "pred_positive_rate": float((tp + fp) / max(1, tp + fn + fp + tn)),
                }
            )
        if threshold_scan_summary:
            best = max(
                threshold_scan_summary,
                key=lambda x: (
                    float(x["f1"]),
                    float(x["precision"]),
                    -abs(float(x["threshold"]) - float(prob_threshold)),
                ),
            )
            recommended_threshold = float(best["threshold"])

    W = 128  # table width
    print("\n" + "=" * W)
    print(f"\U0001f3c6  FINAL EVALUATION REPORT ({eval_phase} SET)")
    print(f"   Session           : {selected_session}")
    print(f"   Pairing mode      : {pairing_mode}")
    print(f"   Server checkpoint : {Path(server_path).name}  (round={server_round})")
    print(f"   Eval cfg source   : {eval_cfg_source}")
    if forced_prob_threshold is not None:
        print(f"   Forced threshold  : {forced_prob_threshold:.2f}")
    if recommended_threshold is not None:
        print(f"   Scan best threshold (by F1): {recommended_threshold:.2f}")
    print("=" * W)
    hdr = (f"  {'Client':<8} {'BestRound':>10} {'TrainLoss':>10}"
           f"  {'Samples':>8}  {'MSE':>8}  {'MAE':>8}  {'ClsAcc':>8}  {'ClsF1':>8}"
           f"  {'PR-AUC':>8}  {'ROC-AUC':>8}  {'Brier':>8}  {'OpF1':>8}")
    sep = (f"  {'──────':<8} {'─────────':>10} {'─────────':>10}"
           f"  {'───────':>8}  {'───────':>8}  {'───────':>8}  {'──────':>8}  {'──────':>8}"
           f"  {'──────':>8}  {'───────':>8}  {'──────':>8}  {'──────':>8}")
    print(hdr)
    print(sep)
    for r in all_results:
        br  = str(r['best_round'])
        tl  = f"{r['train_loss']:.4f}" if not np.isnan(r['train_loss']) else "N/A"
        print(f"  {r['client_id']:<8} {br:>10} {tl:>10}"
              f"  {r['samples']:>8,}  {r['mse']:>8.4f}  {r['mae']:>8.4f}  {r['accuracy']:>7.2f}%  {r['f1']:>8.4f}"
              f"  {r['auprc']:>8.4f}  {r['roc_auc']:>8.4f}  {r['brier']:>8.4f}  {r['op_f1']:>8.4f}")

    if len(all_results) > 1:
        avg_mse = np.mean([r["mse"]      for r in all_results])
        avg_mae = np.mean([r["mae"]      for r in all_results])
        avg_acc = np.mean([r["accuracy"] for r in all_results])
        avg_f1  = np.mean([r["f1"] for r in all_results])
        avg_auprc = np.mean([r["auprc"] for r in all_results])
        avg_roc_auc = np.mean([r["roc_auc"] for r in all_results])
        avg_brier = np.mean([r["brier"] for r in all_results])
        avg_op_f1 = np.mean([r["op_f1"] for r in all_results])
        tot     = sum(    r["samples"]   for r in all_results)
        print(sep)
        print(f"  {'AVERAGE':<8} {'':>10} {'':>10}"
              f"  {tot:>8,}  {avg_mse:>8.4f}  {avg_mae:>8.4f}  {avg_acc:>7.2f}%  {avg_f1:>8.4f}"
              f"  {avg_auprc:>8.4f}  {avg_roc_auc:>8.4f}  {avg_brier:>8.4f}  {avg_op_f1:>8.4f}")
    print("=" * W + "\n")

    if threshold_scan_summary:
        print("Threshold scan summary (weighted/global confusion):")
        print("  threshold  f1      precision  recall    pred_pos")
        for row in threshold_scan_summary:
            print(
                f"  {row['threshold']:>8.2f}  {row['f1']:>6.4f}  {row['precision']:>9.4f}  "
                f"{row['recall']:>7.4f}  {row['pred_positive_rate']:>8.4f}"
            )

    bw_session_dir = project_root / "bestweights" / selected_session
    try:
        rel_path = Path(server_path).relative_to(bw_session_dir)
        scenario_part = rel_path.parts[0] if len(rel_path.parts) > 1 and rel_path.parts[0] != "periodic" else ""
    except ValueError:
        scenario_part = ""

    save_dir = project_root / "results" / selected_session
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = save_dir / scenario_part if scenario_part else save_dir

    if args.scenario:
        report_stem = f"{args.scenario}_eval_report"
    elif scenario_part:
        report_stem = f"{scenario_part}_eval_report"
    else:
        safe_session_str = str(selected_session).replace("/", "_").replace("\\", "_")
        report_stem = f"{safe_session_str}_eval_report"
    
    if report_tag:
        report_stem = f"{report_stem}_{report_tag}"

    import re
    training_meta_files = list(results_dir.glob("*_meta.json"))
    client_telemetry = {}
    for mf in training_meta_files:
        if "progress" in mf.name:
            continue
        try:
            m = re.search(r'client(\d+)', mf.name)
            if m:
                with open(mf, "r") as f:
                    client_telemetry[int(m.group(1))] = json.load(f)
        except Exception:
            pass

    for r in all_results:
        cid = r["client_id"]
        tm = client_telemetry.get(cid, {})
        r["cpu_percent"] = round(float(tm["avg_cpu_percent"]), 1) if tm.get("avg_cpu_percent") is not None else np.nan
        r["mem_percent"] = round(float(tm["avg_mem_percent"]), 1) if tm.get("avg_mem_percent") is not None else np.nan
        r["runtime_s"] = round(float(tm["total_runtime_s"]), 1) if tm.get("total_runtime_s") is not None else np.nan

        msb = tm.get("model_size_bytes")
        r["model_size_mb"] = round(msb / (1024 * 1024), 2) if msb is not None else np.nan
        
        apb = tm.get("avg_payload_bytes")
        r["payload_bytes"] = round(float(apb), 1) if apb is not None else np.nan
        
        alm = tm.get("avg_latency_ms")
        r["latency_ms"] = round(float(alm), 1) if alm is not None else np.nan
        
        rt = tm.get("total_runtime_s")
        r["throughput_sps"] = round(r.get("samples", 0) / rt, 1) if rt and float(rt) > 0 else np.nan
        r["net_sent_mb"] = tm.get("net_sent_mb", np.nan)
        r["net_recv_mb"] = tm.get("net_recv_mb", np.nan)
        r["mem_peak_mb"] = tm.get("mem_peak_mb", np.nan)
        r["sync_bytes_sent_mb"] = tm.get("sync_bytes_sent_mb", np.nan)
        r["sync_bytes_recv_mb"] = tm.get("sync_bytes_recv_mb", np.nan)
        # 數據量吞吐量: (平均每步傳輸量 × 總步數) / 訓練時間 (KB/s)
        num_records = tm.get("num_records")
        if apb and num_records and rt and float(rt) > 0:
            r["data_throughput_kbps"] = round(
                (float(apb) * int(num_records) / 1024) / float(rt), 2
            )
        else:
            r["data_throughput_kbps"] = np.nan
        # 總傳輸量 (MB)：全訓練過程所有 Forward Pass 的 Payload 合計
        # 對應 Results Table 的 Traff (payload) 欄位
        if apb and num_records:
            r["total_payload_mb"] = round(float(apb) * int(num_records) / (1024 * 1024), 4)
        else:
            r["total_payload_mb"] = np.nan

    # 🚀 CALCULATE AGGREGATED TELEMETRY & WEIGHTED METRICS (Move up for CSV inclusion)
    total_samples = float(sum(r["samples"] for r in all_results))
    weighted = {
        "mse": float(sum(r["mse"] * r["samples"] for r in all_results) / total_samples),
        "mae": float(sum(r["mae"] * r["samples"] for r in all_results) / total_samples),
        "accuracy": float(sum(r["accuracy"] * r["samples"] for r in all_results) / total_samples),
        "f1": float(sum(r["f1"] * r["samples"] for r in all_results) / total_samples),
        "auprc": float(sum(r["auprc"] * r["samples"] for r in all_results) / total_samples),
        "roc_auc": float(sum(r["roc_auc"] * r["samples"] for r in all_results) / total_samples),
        "brier": float(sum(r["brier"] * r["samples"] for r in all_results) / total_samples),
        "positive_rate": float(sum(r["positive_rate"] * r["samples"] for r in all_results) / total_samples),
        "pred_positive_rate": float(sum(r["pred_positive_rate"] * r["samples"] for r in all_results) / total_samples),
        "tp": int(sum(r["tp"] for r in all_results)),
        "fn": int(sum(r["fn"] for r in all_results)),
        "fp": int(sum(r["fp"] for r in all_results)),
        "tn": int(sum(r["tn"] for r in all_results)),
        "rain_mse": float(np.nanmean([r.get("rain_mse", float("nan")) for r in all_results])),
        "rain_mae": float(np.nanmean([r.get("rain_mae", float("nan")) for r in all_results])),
    }

    # Fetch and aggregate hardware telemetry
    sys_telemetry = {}
    if training_meta_files:
        try:
            runtimes, cpus, mems = [], [], []
            net_sent, net_recv, mem_peaks = [], [], []
            sync_sent, sync_recv = [], []
            for mf in training_meta_files:
                if "progress" in mf.name: continue
                with open(mf, "r") as f:
                    mt = json.load(f)
                    if mt.get("total_runtime_s"): runtimes.append(float(mt["total_runtime_s"]))
                    if mt.get("avg_cpu_percent"): cpus.append(float(mt["avg_cpu_percent"]))
                    if mt.get("avg_mem_percent"): mems.append(float(mt["avg_mem_percent"]))
                    if mt.get("net_sent_mb"): net_sent.append(float(mt["net_sent_mb"]))
                    if mt.get("net_recv_mb"): net_recv.append(float(mt["net_recv_mb"]))
                    if mt.get("mem_peak_mb"): mem_peaks.append(float(mt["mem_peak_mb"]))
                    if mt.get("sync_bytes_sent_mb"): sync_sent.append(float(mt["sync_bytes_sent_mb"]))
                    if mt.get("sync_bytes_recv_mb"): sync_recv.append(float(mt["sync_bytes_recv_mb"]))
            
            if runtimes: sys_telemetry["avg_runtime_s"] = round(sum(runtimes)/len(runtimes), 2)
            if cpus: sys_telemetry["avg_cpu_percent"] = round(sum(cpus)/len(cpus), 1)
            if mems: sys_telemetry["avg_mem_percent"] = round(sum(mems)/len(mems), 1)
            if net_sent: sys_telemetry["total_net_sent_mb"] = round(sum(net_sent), 2)
            if net_recv: sys_telemetry["total_net_recv_mb"] = round(sum(net_recv), 2)
            if mem_peaks: sys_telemetry["avg_mem_peak_mb"] = round(sum(mem_peaks)/len(mem_peaks), 2)
            if sync_sent: sys_telemetry["total_sync_sent_mb"] = round(sum(sync_sent), 4)
            if sync_recv: sys_telemetry["total_sync_recv_mb"] = round(sum(sync_recv), 4)
            if sys_telemetry.get("avg_runtime_s", 0) > 0:
                sys_telemetry["throughput_sps"] = round(total_samples / sys_telemetry["avg_runtime_s"], 2)
                # 系統級數據量吞吐量：所有 Client 傳輸 Bytes 總和 / 平均訓練時間
                total_payload_bytes = sum(
                    float(r.get("payload_bytes", 0) or 0)
                    * float(client_telemetry.get(cid, {}).get("num_records", 0) or 0)
                    for cid, r in zip(
                        [res.get("client_id") for res in all_results],
                        all_results
                    )
                )
                if total_payload_bytes > 0:
                    sys_telemetry["data_throughput_kbps"] = round(
                        (total_payload_bytes / 1024) / sys_telemetry["avg_runtime_s"], 2
                    )
        except Exception:
            pass

    report_csv = save_dir / f"{report_stem}.csv"
    
    # 重新整理每一行的順序，確保 monthly_details 在最後
    csv_rows = []
    for r in all_results:
        # 先抓取所有非 monthly_details 且非 threshold_scan 的欄位
        row = {k: v for k, v in r.items() if k not in ["threshold_scan", "monthly_details"]}
        # 最後補上 monthly_details
        row["monthly_details"] = r.get("monthly_details")
        csv_rows.append(row)

    # 準備 Summary 行
    summary_row = {
        "client_id": "SUMMARY",
        "samples": int(total_samples),
        "mse": weighted["mse"],
        "mae": weighted["mae"],
        "rain_mse": weighted["rain_mse"],
        "rain_mae": weighted["rain_mae"],
        "accuracy": weighted["accuracy"],
        "f1": weighted["f1"],
        "auprc": weighted["auprc"],
        "roc_auc": weighted["roc_auc"],
        "brier": weighted["brier"],
        "tp": weighted["tp"],
        "fn": weighted["fn"],
        "fp": weighted["fp"],
        "tn": weighted["tn"],
        "cpu_percent": sys_telemetry.get("avg_cpu_percent", np.nan),
        "mem_percent": sys_telemetry.get("avg_mem_percent", np.nan),
        "runtime_s": sys_telemetry.get("avg_runtime_s", np.nan),
        "throughput_sps": sys_telemetry.get("throughput_sps", np.nan),
        "data_throughput_kbps": sys_telemetry.get("data_throughput_kbps", np.nan),
        "total_payload_mb": round(sum(r.get("total_payload_mb", 0) or 0 for r in all_results), 4),
        "net_sent_mb": sys_telemetry.get("total_net_sent_mb", np.nan),
        "net_recv_mb": sys_telemetry.get("total_net_recv_mb", np.nan),
        "mem_peak_mb": sys_telemetry.get("avg_mem_peak_mb", np.nan),
        "sync_bytes_sent_mb": sys_telemetry.get("total_sync_sent_mb", np.nan),
        "sync_bytes_recv_mb": sys_telemetry.get("total_sync_recv_mb", np.nan),
        "monthly_details": "" # Summary 行留空
    }
    csv_rows.append(summary_row)
    
    # 直接儲存 (Pandas 會遵循第一個 dict 的 keys 順序)
    pd.DataFrame(csv_rows).to_csv(report_csv, index=False)

    report_json = save_dir / f"{report_stem}.json"
    summary = {
        "session": selected_session,
        "pairing_mode": pairing_mode,
        "server_checkpoint": Path(server_path).name,
        "server_round": server_round,
        "report_tag": report_tag,
        "strict_pairing_default": True,
        "allow_latest_best_fallback": bool(args.allow_latest_best_fallback),
        "prefer_checkpoint_threshold": bool(args.prefer_checkpoint_threshold),
        "forced_prob_threshold": forced_prob_threshold,
        "device": str(device),
        "eval_phase": eval_phase,
        "split_date": str(split_date),
        "eval_max_samples_per_sensor": eval_max_samples,
        "eval_config_source": eval_cfg_source,
        "eval_config": {
            "num_clients": num_clients,
            "processed_dir": processed_dir,
            "feature_cols": active_features,
            "seq_len": seq_len,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "server_head_width": head_width,
            "server_head_dropout": head_dropout,
            "rain_probability_threshold": (
                forced_prob_threshold if forced_prob_threshold is not None else prob_threshold
            ),
            "rain_probability_threshold_source": (
                "forced_cli"
                if forced_prob_threshold is not None
                else ("client_checkpoint" if args.prefer_checkpoint_threshold else "config")
            ),
            "rain_threshold_mm": rain_threshold,
            "target_transform": target_mode,
        },
        "num_clients_evaluated": len(all_results),
        "weighted_overall": weighted,
        "threshold_scan_summary": threshold_scan_summary,
        "recommended_threshold_by_f1": recommended_threshold,
        "clients": all_results,
    }

    # 🚀 Inject Training Phase Metadata into Summary
    if sys_telemetry:
        summary.update({
            "training_total_runtime_s": sys_telemetry.get("avg_runtime_s"),
            "training_avg_cpu_percent": sys_telemetry.get("avg_cpu_percent"),
            "training_avg_mem_percent": sys_telemetry.get("avg_mem_percent"),
            "training_throughput_samples_s": sys_telemetry.get("throughput_sps"),
        })
        
        print("\n" + "=" * W)
        print(f"🌡️  HARDWARE TELEMETRY & EFFICIENCY")
        print("=" * W)
        print(f"   Total Runtime      : {sys_telemetry.get('avg_runtime_s', 'N/A')} s")
        print(f"   System Throughput  : {sys_telemetry.get('throughput_sps', 'N/A')} samples/s")
        print(f"   Average CPU Usage  : {sys_telemetry.get('avg_cpu_percent', 'N/A')} %")
        print(f"   Average Memory     : {sys_telemetry.get('avg_mem_percent', 'N/A')} %")
        print("=" * W + "\n")
    else:
        print(f"[WARN] No training metadata found in {results_dir}")


    with open(report_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Saved evaluation CSV : {report_csv}")
    print(f"[INFO] Saved evaluation JSON: {report_json}")


if __name__ == "__main__":
    evaluate()
