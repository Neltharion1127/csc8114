import numpy as np
import pandas as pd

from src.shared.common import cfg
from src.shared.targets import is_rain

FUTURE_RAIN_COL = "future_rain"
LEGACY_FUTURE_RAIN_COL = "future_24h_rain"


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
    raw = horizon if horizon is not None else cfg.get("model", {}).get("horizon", 24)
    try:
        resolved = int(raw)
    except (TypeError, ValueError):
        resolved = 24
    return max(1, resolved)


def partition_client_files(
    all_files: list[str],
    *,
    client_id: int,
    num_clients: int,
) -> list[str]:
    """Assign exactly one file per client (strict 1:1 mapping).

    Files are sorted deterministically; client N receives sorted_files[N-1].
    Raises ValueError if num_clients exceeds the number of available files,
    since that would leave some clients with no data.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be positive, got {num_clients}")
    if client_id <= 0:
        raise ValueError(f"client_id must be positive, got {client_id}")

    sorted_files = sorted(all_files)
    if num_clients > len(sorted_files):
        raise ValueError(
            f"num_clients ({num_clients}) exceeds available sensor files "
            f"({len(sorted_files)}). Reduce num_clients or add more sensor files."
        )

    return [sorted_files[client_id - 1]]


def resolve_split_pos(df: pd.DataFrame, split_date: pd.Timestamp) -> int:
    """Resolve split position robustly even when split_date is not in index."""
    try:
        split_pos = df.index.get_indexer([split_date], method="pad")[0]
    except Exception:
        split_pos = int(len(df) * 0.8)
    return int(split_pos)


def get_dataset_split(ts: pd.Timestamp) -> str:
    """Chronological time-based split using absolute date boundaries from config."""
    data_cfg = cfg.get("data", {})
    train_end = pd.Timestamp(data_cfg.get("train_end", "2024-12-31"))
    val_end   = pd.Timestamp(data_cfg.get("val_end",   "2025-06-30"))
    if ts < train_end:
        return "TRAIN"
    if ts < val_end:
        return "VAL"
    return "TEST"


def collect_eval_indices(
    df: pd.DataFrame,
    *,
    target_phase: str,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect indices belonging to a given monthly split phase (TRAIN/VAL/TEST)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Monthly cycle splitting requires a DatetimeIndex.")
    horizon = resolve_horizon(horizon)
    all_indices = np.arange(len(df))
    mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)

    timestamps = pd.to_datetime(df.index)
    phase_mask = np.array([get_dataset_split(ts) == target_phase for ts in timestamps])
    mask = mask & phase_mask

    if FUTURE_RAIN_COL in df.columns:
        mask = mask & df[FUTURE_RAIN_COL].notna().to_numpy()
    elif LEGACY_FUTURE_RAIN_COL in df.columns:
        mask = mask & df[LEGACY_FUTURE_RAIN_COL].notna().to_numpy()

    return all_indices[mask]


def collect_eval_indices_capped(
    df: pd.DataFrame,
    *,
    target_phase: str,
    eval_max_samples: int = 0,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect eval indices for a monthly phase, with optional fixed cap."""
    eval_indices = collect_eval_indices(
        df,
        target_phase=target_phase,
        min_history=min_history,
        horizon=horizon,
    )
    if eval_max_samples > 0 and len(eval_indices) > eval_max_samples:
        picks = np.linspace(0, len(eval_indices) - 1, eval_max_samples, dtype=int)
        eval_indices = eval_indices[picks]
    return eval_indices


def collect_test_indices(
    df: pd.DataFrame,
    *,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect TEST-phase indices."""
    return collect_eval_indices(df, target_phase="TEST", min_history=min_history, horizon=horizon)


def collect_test_indices_capped(
    df: pd.DataFrame,
    *,
    eval_max_samples: int = 0,
    min_history: int = 24,
    horizon: int | None = None,
) -> np.ndarray:
    """Collect TEST-phase indices with optional cap."""
    return collect_eval_indices_capped(
        df,
        target_phase="TEST",
        eval_max_samples=eval_max_samples,
        min_history=min_history,
        horizon=horizon,
    )


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
    all_indices = np.arange(len(df))

    # Monthly cycle logic
    target_phase = "TRAIN" if is_training else "TEST"
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Predicting precipitation requires a DatetimeIndex.")
    
    timestamps = pd.to_datetime(df.index)
    valid_mask = (all_indices >= min_history) & (all_indices < len(df) - horizon)
    
    # Filter indices that match the target phase
    possible_indices = all_indices[valid_mask]
    phase_compliant = [idx for idx in possible_indices if get_dataset_split(timestamps[idx]) == target_phase]
    eligible_indices = np.array(phase_compliant)
    
    if len(eligible_indices) == 0:
        return None

    target_col = _resolve_target_col(df)
    target_arr = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    valid_target = np.isfinite(target_arr)
    rain_mask = np.zeros_like(valid_target, dtype=bool)
    rain_mask[valid_target] = np.array(
        [is_rain(v, threshold=rain_threshold) for v in target_arr[valid_target]],
        dtype=bool,
    )
    dry_mask = valid_target & ~rain_mask

    # Intersect monthly-eligible indices with data-quality masks
    rainy_pos = np.intersect1d(eligible_indices, np.where(rain_mask)[0])
    dry_pos = np.intersect1d(eligible_indices, np.where(dry_mask)[0])

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
