import numpy as np
import pandas as pd


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
) -> tuple[int, str] | None:
    """Pick a balanced rainy/dry sample index for train or test."""
    try:
        split_pos = df.index.get_indexer([split_date], method="pad")[0]
    except Exception:
        split_pos = int(len(df) * 0.8)

    all_indices = np.arange(len(df))
    base_mask = (all_indices >= 24) & (all_indices < len(df) - 3)

    if is_training:
        base_mask = base_mask & (all_indices < split_pos)
    else:
        base_mask = base_mask & (all_indices >= split_pos)

    rainy_pos = all_indices[base_mask & (df["future_3h_rain"] > 0)]
    dry_pos = all_indices[base_mask & (df["future_3h_rain"] == 0)]

    if len(rainy_pos) > 0 and np.random.rand() > 0.5:
        return int(np.random.choice(rainy_pos)), "RAIN_SAMPLE"
    if len(dry_pos) > 0:
        return int(np.random.choice(dry_pos)), "DRY_SAMPLE"
    return None
