import numpy as np
import pandas as pd


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
) -> tuple[int, str] | None:
    """Pick one train/test sample with optional rain oversampling."""
    split_pos = resolve_split_pos(df, split_date)

    all_indices = np.arange(len(df))
    base_mask = (all_indices >= 24) & (all_indices < len(df) - 3)

    if is_training:
        base_mask = base_mask & (all_indices < split_pos)
    else:
        base_mask = base_mask & (all_indices >= split_pos)

    rainy_pos = all_indices[base_mask & (df["future_3h_rain"] > 0)]
    dry_pos = all_indices[base_mask & (df["future_3h_rain"] == 0)]

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
