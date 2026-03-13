import numpy as np
import pandas as pd

from src.shared.targets import is_rain


def partition_client_files(
    all_files: list[str],
    *,
    client_id: int,
    num_clients: int,
) -> list[str]:
    """Partition file list across clients using the same deterministic rule as training."""
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if client_id <= 0:
        raise ValueError(f"client_id must be positive, got {client_id}")

    sorted_files = sorted(all_files)
    chunk_size = len(sorted_files) // num_clients
    start_idx = (client_id - 1) * chunk_size
    end_idx = start_idx + chunk_size if client_id < num_clients else len(sorted_files)
    return sorted_files[start_idx:end_idx]


def resolve_split_pos(df: pd.DataFrame, split_date: pd.Timestamp) -> int:
    """Resolve split position robustly even when split_date is not in index."""
    try:
        split_pos = df.index.get_indexer([split_date], method="pad")[0]
    except Exception:
        split_pos = int(len(df) * 0.8)
    return int(split_pos)


def collect_test_indices(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    min_history: int = 24,
    horizon: int = 3,
) -> np.ndarray:
    """Collect deterministic test indices with valid context and non-NaN targets."""
    split_pos = resolve_split_pos(df, split_date)
    all_indices = np.arange(len(df))
    base_mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)
    test_mask = base_mask & (all_indices >= split_pos)
    if "future_3h_rain" in df.columns:
        valid_target = df["future_3h_rain"].notna().to_numpy()
        test_mask = test_mask & valid_target
    return all_indices[test_mask]


def collect_test_indices_capped(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    eval_max_samples: int = 0,
    min_history: int = 24,
    horizon: int = 3,
) -> np.ndarray:
    """Collect deterministic test indices, with optional fixed-size cap per sensor."""
    test_indices = collect_test_indices(
        df,
        split_date,
        min_history=min_history,
        horizon=horizon,
    )
    if eval_max_samples > 0 and len(test_indices) > eval_max_samples:
        picks = np.linspace(0, len(test_indices) - 1, eval_max_samples, dtype=int)
        test_indices = test_indices[picks]
    return test_indices


def load_sensor_data(file_path: str) -> pd.DataFrame:
    """
    Reads a single sensor parquet file, ensures a DatetimeIndex,
    and computes the future rainfall target column.
    """
    df = pd.read_parquet(file_path)
    if "Timestamp" in df.columns:
        df.set_index("Timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)

    df["future_3h_rain"] = df["Rain"].shift(-3).rolling(window=3).sum()
    return df


def sample_index(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    is_training: bool = True,
    rain_sample_ratio: float | None = None,
    min_history: int = 24,
    horizon: int = 3,
    rain_threshold: float | None = None,
) -> tuple[int, str] | None:
    """Pick one train/test sample with optional rain oversampling."""
    split_pos = resolve_split_pos(df, split_date)

    all_indices = np.arange(len(df))
    base_mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)

    if is_training:
        base_mask = base_mask & (all_indices < split_pos)
    else:
        base_mask = base_mask & (all_indices >= split_pos)

    target_arr = pd.to_numeric(df["future_3h_rain"], errors="coerce").to_numpy(dtype=float)
    valid_target = np.isfinite(target_arr)
    rain_mask = np.zeros_like(valid_target, dtype=bool)
    rain_mask[valid_target] = np.array(
        [is_rain(v, threshold=rain_threshold) for v in target_arr[valid_target]],
        dtype=bool,
    )
    dry_mask = valid_target & ~rain_mask

    rainy_pos = all_indices[base_mask & rain_mask]
    dry_pos = all_indices[base_mask & dry_mask]

    if rain_sample_ratio is None:
        rain_sample_ratio = 0.5 if is_training else 0.0
    rain_sample_ratio = float(np.clip(rain_sample_ratio, 0.0, 1.0))

    if len(rainy_pos) > 0 and np.random.rand() < rain_sample_ratio:
        return int(np.random.choice(rainy_pos)), "RAIN_SAMPLE"
    if len(dry_pos) > 0:
        return int(np.random.choice(dry_pos)), "DRY_SAMPLE"
    if len(rainy_pos) > 0:
        return int(np.random.choice(rainy_pos)), "RAIN_SAMPLE"
    return None
