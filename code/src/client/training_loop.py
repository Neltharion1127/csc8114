from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.client.data_pipeline import collect_eval_indices_capped, load_sensor_data, sample_index
from src.client.forward_step import run_forward_step
from src.client.scheduler_state import SchedulerState
from src.shared.targets import is_rain, rain_probability_threshold


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1 = float(2.0 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    total = tp + fn + fp + tn
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
    }


def _select_best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    default_threshold: float,
) -> tuple[float, dict[str, float | int], dict[str, float | int]]:
    default_pred = (probs >= default_threshold).astype(np.int32)
    default_metrics = _binary_metrics(y_true, default_pred)

    best_threshold = float(default_threshold)
    best_metrics = default_metrics
    best_score = (
        float(default_metrics["f1"]),
        float(default_metrics["precision"]),
        -abs(float(default_threshold) - float(default_threshold)),
    )
    for threshold in np.linspace(0.0, 1.0, 201):
        y_pred = (probs >= threshold).astype(np.int32)
        metrics = _binary_metrics(y_true, y_pred)
        score = (
            float(metrics["f1"]),
            float(metrics["precision"]),
            -abs(float(threshold) - float(default_threshold)),
        )
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics, default_metrics


def preload_sensor_data(client_id: int, client_files: list[str]) -> dict[str, pd.DataFrame]:
    print(f"[CLIENT {client_id}] Pre-loading sensor data into memory...")
    sensor_data_cache: dict[str, pd.DataFrame] = {}
    for file_path in client_files:
        sensor_id = Path(file_path).stem
        try:
            sensor_data_cache[file_path] = load_sensor_data(file_path)
        except Exception as e:
            print(f"[CLIENT {client_id} ERROR] Failed to load {sensor_id}: {e}")
    if not sensor_data_cache:
        raise RuntimeError(
            f"Client {client_id} could not load any sensor data from {len(client_files)} assigned files. "
            "Check dataset contents and preprocessing outputs."
        )
    return sensor_data_cache


def compute_feature_stats(
    *,
    client_id: int,
    sensor_data_cache: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    print(f"[CLIENT {client_id}] Calculating feature statistics for normalization...")
    all_combined = pd.concat(sensor_data_cache.values())
    feat_mean = all_combined[feature_cols].mean().values
    feat_std = all_combined[feature_cols].std().values + 1e-9
    return feat_mean, feat_std


def build_eval_index_cache(
    *,
    client_id: int,
    sensor_data_cache: dict[str, pd.DataFrame],
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    eval_max_samples: int,
    seq_len: int,
    label: str,
    horizon: int = 3,
) -> tuple[dict[str, np.ndarray], int, int]:
    eval_index_cache: dict[str, np.ndarray] = {}
    total_eval_samples = 0
    total_eval_positive = 0
    for file_path, df in sensor_data_cache.items():
        eval_indices = collect_eval_indices_capped(
            df,
            start_date=start_date,
            end_date=end_date,
            eval_max_samples=eval_max_samples,
            min_history=seq_len,
            horizon=horizon,
        )
        eval_index_cache[file_path] = eval_indices
        total_eval_samples += int(len(eval_indices))
        if len(eval_indices) > 0:
            total_eval_positive += int(df["future_3h_rain"].iloc[eval_indices].apply(is_rain).sum())
    print(
        f"[CLIENT {client_id}] Fixed {label.lower()} set prepared: "
        f"samples={total_eval_samples} positives={total_eval_positive} "
        f"(per_sensor_cap={eval_max_samples if eval_max_samples > 0 else 'FULL'})"
    )
    return eval_index_cache, total_eval_samples, total_eval_positive


def run_train_epoch(
    *,
    stub,
    client_id: int,
    client_model,
    optimizer,
    client_files: list[str],
    sensor_data_cache: dict[str, pd.DataFrame],
    split_date: pd.Timestamp,
    train_state: SchedulerState,
    feature_cols: list[str],
    feat_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    local_steps: int,
    rain_sample_ratio: float,
    seq_len: int,
    epoch: int,
    experimental_logs: list[dict],
    epoch_logs: list[dict],
    horizon: int = 3,
    rain_threshold: float | None = None,
) -> int:
    epoch_train_steps = 0
    for file_path in client_files:
        sensor_id = Path(file_path).stem
        try:
            df = sensor_data_cache.get(file_path)
            if df is None:
                continue
            for _ in range(local_steps):
                optimizer.zero_grad()
                result = sample_index(
                    df,
                    split_date,
                    is_training=True,
                    rain_sample_ratio=rain_sample_ratio,
                    min_history=seq_len,
                    horizon=horizon,
                    rain_threshold=rain_threshold,
                )
                if result is None:
                    continue
                target_idx, mode = result
                target_value = float(df["future_3h_rain"].iloc[target_idx])
                log_entry = run_forward_step(
                    stub,
                    client_id,
                    client_model,
                    optimizer,
                    df,
                    target_idx,
                    target_value,
                    mode,
                    sensor_id,
                    train_state.compression_mode,
                    feature_cols,
                    feat_stats,
                    device,
                    is_training=True,
                    last_latency_ms=train_state.last_latency_ms,
                    seq_len=seq_len,
                )
                train_state.update(log_entry)
                epoch_train_steps += 1
                epoch_record = {"Epoch": epoch + 1, "Status": "TRAIN", "Sensor": sensor_id, **log_entry}
                experimental_logs.append(epoch_record)
                epoch_logs.append(epoch_record)
        except Exception as e:
            print(f"[CLIENT {client_id} ERROR] {sensor_id}: {e}")
    return epoch_train_steps


def run_eval_epoch(
    *,
    stub,
    client_id: int,
    client_model,
    optimizer,
    client_files: list[str],
    sensor_data_cache: dict[str, pd.DataFrame],
    eval_index_cache: dict[str, np.ndarray],
    eval_state: SchedulerState,
    feature_cols: list[str],
    feat_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    seq_len: int,
    epoch: int,
    experimental_logs: list[dict],
    epoch_logs: list[dict],
    phase_label: str = "VAL",
) -> tuple[list[float], dict[str, float | int]]:
    epoch_eval_losses: list[float] = []
    eval_targets: list[float] = []
    eval_probs: list[float] = []
    default_threshold = float(rain_probability_threshold())
    with torch.no_grad():
        for file_path in client_files:
            sensor_id = Path(file_path).stem
            try:
                df = sensor_data_cache.get(file_path)
                if df is None:
                    continue
                eval_indices = eval_index_cache.get(file_path)
                if eval_indices is None or len(eval_indices) == 0:
                    continue
                for target_idx in eval_indices:
                    target_value = float(df["future_3h_rain"].iloc[target_idx])
                    log_entry = run_forward_step(
                        stub,
                        client_id,
                        client_model,
                        optimizer,
                        df,
                        int(target_idx),
                        target_value,
                        f"FIXED_{phase_label}",
                        sensor_id,
                        eval_state.compression_mode,
                        feature_cols,
                        feat_stats,
                        device,
                        is_training=False,
                        last_latency_ms=eval_state.last_latency_ms,
                        seq_len=seq_len,
                    )
                    eval_state.update(log_entry)
                    epoch_record = {"Epoch": epoch + 1, "Status": phase_label, "Sensor": sensor_id, **log_entry}
                    experimental_logs.append(epoch_record)
                    epoch_logs.append(epoch_record)
                    if log_entry["Loss"] is not None:
                        epoch_eval_losses.append(float(log_entry["Loss"]))
                    eval_targets.append(float(log_entry["Target"]))
                    eval_probs.append(float(log_entry.get("RainProbability", 0.0)))
            except Exception as e:
                print(f"[CLIENT {client_id} WARN] Eval failed on {sensor_id}: {e}")

    if not eval_targets:
        empty_metrics = {
            "tp": 0,
            "fn": 0,
            "fp": 0,
            "tn": 0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "selected_threshold": default_threshold,
            "default_threshold": default_threshold,
            "default_recall": 0.0,
            "default_precision": 0.0,
            "default_f1": 0.0,
            "default_accuracy": 0.0,
        }
        return epoch_eval_losses, empty_metrics

    y_true = np.array([1 if is_rain(target) else 0 for target in eval_targets], dtype=np.int32)
    probs_arr = np.array(eval_probs, dtype=np.float32)
    selected_threshold, selected_metrics, default_metrics = _select_best_threshold(
        y_true,
        probs_arr,
        default_threshold=default_threshold,
    )
    eval_metrics: dict[str, float | int] = {
        **selected_metrics,
        "phase": phase_label,
        "selected_threshold": float(selected_threshold),
        "default_threshold": default_threshold,
        "default_recall": float(default_metrics["recall"]),
        "default_precision": float(default_metrics["precision"]),
        "default_f1": float(default_metrics["f1"]),
        "default_accuracy": float(default_metrics["accuracy"]),
    }
    return epoch_eval_losses, eval_metrics
