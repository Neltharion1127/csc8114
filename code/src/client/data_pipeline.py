import numpy as np
import pandas as pd

from src.shared.common import cfg
from src.shared.targets import is_rain

FUTURE_RAIN_COL = "future_rain"
LEGACY_FUTURE_RAIN_COL = "future_3h_rain"


def _resolve_target_col(df: pd.DataFrame) -> str:
    """Return the preferred target column, with legacy fallback."""
    if FUTURE_RAIN_COL in df.columns:
        return FUTURE_RAIN_COL
    if LEGACY_FUTURE_RAIN_COL in df.columns:
        return LEGACY_FUTURE_RAIN_COL
    raise KeyError(
        f"Missing target column. Expected '{FUTURE_RAIN_COL}' "
        f"(or legacy '{LEGACY_FUTURE_RAIN_COL}')."
    )


def resolve_horizon(horizon: int | None = None) -> int:
    """Resolve prediction horizon from arg or config, clamped to >=1."""
    raw = horizon if horizon is not None else cfg.get("model", {}).get("horizon", 3)
    try:
        resolved = int(raw)
    except (TypeError, ValueError):
        resolved = 3
    return max(1, resolved)


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


def collect_eval_indices(
    df: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect deterministic indices in a [start_date, end_date) time window."""
    horizon = resolve_horizon(horizon)

    def _to_naive_timestamp(value: pd.Timestamp | None) -> pd.Timestamp | None:
        if value is None:
            return None
        ts = pd.Timestamp(value)
        return ts.tz_convert(None) if ts.tzinfo is not None else ts

    all_indices = np.arange(len(df))
    mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)

    if isinstance(df.index, pd.DatetimeIndex):
        timestamps = pd.to_datetime(df.index)
        if timestamps.tz is not None:
            timestamps = timestamps.tz_convert(None)
        start_ts = _to_naive_timestamp(start_date)
        end_ts = _to_naive_timestamp(end_date)
        if start_ts is not None:
            mask = mask & (timestamps >= start_ts)
        if end_ts is not None:
            # Keep labels fully inside [start, end): target at t uses rain from (t, t+horizon].
            effective_end = end_ts - pd.Timedelta(hours=horizon)
            mask = mask & (timestamps < effective_end)
    else:
        if start_date is not None:
            start_pos = resolve_split_pos(df, _to_naive_timestamp(start_date))
            mask = mask & (all_indices >= start_pos)
        if end_date is not None:
            end_pos = resolve_split_pos(df, _to_naive_timestamp(end_date))
            effective_end_pos = max(0, int(end_pos) - horizon)
            mask = mask & (all_indices < effective_end_pos)

    if FUTURE_RAIN_COL in df.columns:
        valid_target = df[FUTURE_RAIN_COL].notna().to_numpy()
        mask = mask & valid_target
    elif LEGACY_FUTURE_RAIN_COL in df.columns:
        valid_target = df[LEGACY_FUTURE_RAIN_COL].notna().to_numpy()
        mask = mask & valid_target
    return all_indices[mask]


def collect_eval_indices_capped(
    df: pd.DataFrame,
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
    eval_max_samples: int = 0,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect deterministic eval indices in a time window, with optional fixed cap."""
    eval_indices = collect_eval_indices(
        df,
        start_date=start_date,
        end_date=end_date,
        min_history=min_history,
        horizon=horizon,
    )
    if eval_max_samples > 0 and len(eval_indices) > eval_max_samples:
        picks = np.linspace(0, len(eval_indices) - 1, eval_max_samples, dtype=int)
        eval_indices = eval_indices[picks]
    return eval_indices


def collect_test_indices(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect deterministic test indices with valid context and non-NaN targets."""
    return collect_eval_indices(
        df,
        start_date=split_date,
        end_date=None,
        min_history=min_history,
        horizon=horizon,
    )


def collect_test_indices_capped(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    eval_max_samples: int = 0,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect deterministic test indices, with optional fixed-size cap per sensor."""
    test_indices = collect_eval_indices_capped(
        df,
        start_date=split_date,
        end_date=None,
        eval_max_samples=eval_max_samples,
        min_history=min_history,
        horizon=horizon,
    )
    return test_indices


def load_sensor_data(file_path: str, *, horizon: int | None = None) -> pd.DataFrame:
    """
    Reads a single sensor parquet file, ensures a DatetimeIndex,
    and computes the future rainfall target column.
    """
    df = pd.read_parquet(file_path)
    if "Timestamp" in df.columns:
        df.set_index("Timestamp", inplace=True)
        df.index = pd.to_datetime(df.index)

    horizon = resolve_horizon(horizon)
    df[FUTURE_RAIN_COL] = df["Rain"].shift(-horizon).rolling(window=horizon).sum()
    return df


def sample_index(
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    *,
    is_training: bool = True,
    rain_sample_ratio: float | None = None,
    min_history: int = 24,
    horizon: int | None = None,
    rain_threshold: float | None = None,
) -> tuple[int, str] | None:
    """Pick one train/test sample with optional rain oversampling."""
    horizon = resolve_horizon(horizon)
    split_pos = resolve_split_pos(df, split_date)

    all_indices = np.arange(len(df))
    base_mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)

    if is_training:
        # Avoid split leakage: keep entire future target window inside training period.
        train_end_exclusive = max(0, split_pos - horizon)
        base_mask = base_mask & (all_indices < train_end_exclusive)
    else:
        base_mask = base_mask & (all_indices >= split_pos)

    target_col = _resolve_target_col(df)
    target_arr = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
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
