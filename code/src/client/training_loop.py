from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.client.data_pipeline import collect_test_indices_capped, load_sensor_data, sample_index
from src.client.forward_step import run_forward_step
from src.client.scheduler_state import SchedulerState
from src.shared.targets import is_rain, rain_probability_threshold


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


def build_test_index_cache(
    *,
    client_id: int,
    sensor_data_cache: dict[str, pd.DataFrame],
    split_date: pd.Timestamp,
    eval_max_samples: int,
    seq_len: int,
    horizon: int = 3,
) -> tuple[dict[str, np.ndarray], int, int]:
    test_index_cache: dict[str, np.ndarray] = {}
    total_eval_samples = 0
    total_eval_positive = 0
    for file_path, df in sensor_data_cache.items():
        test_indices = collect_test_indices_capped(
            df,
            split_date,
            eval_max_samples=eval_max_samples,
            min_history=seq_len,
            horizon=horizon,
        )
        test_index_cache[file_path] = test_indices
        total_eval_samples += int(len(test_indices))
        if len(test_indices) > 0:
            total_eval_positive += int(df["future_3h_rain"].iloc[test_indices].apply(is_rain).sum())
    print(
        f"[CLIENT {client_id}] Fixed test set prepared: "
        f"samples={total_eval_samples} positives={total_eval_positive} "
        f"(per_sensor_cap={eval_max_samples if eval_max_samples > 0 else 'FULL'})"
    )
    return test_index_cache, total_eval_samples, total_eval_positive


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
    test_index_cache: dict[str, np.ndarray],
    test_state: SchedulerState,
    feature_cols: list[str],
    feat_stats: tuple[np.ndarray, np.ndarray],
    device: torch.device,
    seq_len: int,
    epoch: int,
    experimental_logs: list[dict],
    epoch_logs: list[dict],
) -> tuple[list[float], int, int, int, int]:
    epoch_test_losses: list[float] = []
    tp = fn = fp = tn = 0
    prob_threshold = rain_probability_threshold()
    with torch.no_grad():
        for file_path in client_files:
            sensor_id = Path(file_path).stem
            try:
                df = sensor_data_cache.get(file_path)
                if df is None:
                    continue
                test_indices = test_index_cache.get(file_path)
                if test_indices is None or len(test_indices) == 0:
                    continue
                for target_idx in test_indices:
                    target_value = float(df["future_3h_rain"].iloc[target_idx])
                    log_entry = run_forward_step(
                        stub,
                        client_id,
                        client_model,
                        optimizer,
                        df,
                        int(target_idx),
                        target_value,
                        "FIXED_TEST",
                        sensor_id,
                        test_state.compression_mode,
                        feature_cols,
                        feat_stats,
                        device,
                        is_training=False,
                        last_latency_ms=test_state.last_latency_ms,
                        seq_len=seq_len,
                    )
                    test_state.update(log_entry)
                    epoch_record = {"Epoch": epoch + 1, "Status": "TEST", "Sensor": sensor_id, **log_entry}
                    experimental_logs.append(epoch_record)
                    epoch_logs.append(epoch_record)
                    if log_entry["Loss"] is not None:
                        epoch_test_losses.append(float(log_entry["Loss"]))
                    true_rain = is_rain(float(log_entry["Target"]))
                    pred_rain = float(log_entry.get("RainProbability", 0.0)) >= prob_threshold
                    if true_rain and pred_rain:
                        tp += 1
                    elif true_rain and not pred_rain:
                        fn += 1
                    elif not true_rain and pred_rain:
                        fp += 1
                    else:
                        tn += 1
            except Exception as e:
                print(f"[CLIENT {client_id} WARN] Eval failed on {sensor_id}: {e}")
    return epoch_test_losses, tp, fn, fp, tn
