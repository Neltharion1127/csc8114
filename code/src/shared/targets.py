import math

import torch

from src.shared.common import cfg


def _training_cfg(config: dict | None = None) -> dict:
    source = cfg if config is None else config
    if not isinstance(source, dict):
        return {}
    return source.get("training", {}) if isinstance(source.get("training", {}), dict) else {}


def target_transform_mode(*, config: dict | None = None, mode: str | None = None) -> str:
    """Return the configured target transform mode."""
    if mode is not None:
        return str(mode).strip().lower()
    return str(_training_cfg(config).get("target_transform", "none")).strip().lower()


def transform_target_scalar(
    value: float,
    *,
    config: dict | None = None,
    mode: str | None = None,
) -> float:
    """Map a raw rainfall target into training space."""
    mode = target_transform_mode(config=config, mode=mode)
    value = max(float(value), 0.0)
    if mode == "log1p":
        return math.log1p(value)
    return value


def inverse_target_scalar(
    value: float,
    *,
    config: dict | None = None,
    mode: str | None = None,
) -> float:
    """Map a prediction from training space back into raw rainfall units."""
    mode = target_transform_mode(config=config, mode=mode)
    value = float(value)
    if mode == "log1p":
        return max(math.expm1(value), 0.0)
    return value


def transform_target_tensor(
    tensor: torch.Tensor,
    *,
    config: dict | None = None,
    mode: str | None = None,
) -> torch.Tensor:
    """Map a raw rainfall tensor into training space."""
    mode = target_transform_mode(config=config, mode=mode)
    if mode == "log1p":
        return torch.log1p(torch.clamp(tensor, min=0.0))
    return tensor


def rain_threshold_mm(*, config: dict | None = None) -> float:
    """Rain/no-rain decision threshold in raw rainfall units (mm)."""
    return float(_training_cfg(config).get("rain_threshold_mm", 0.1))


def is_rain(value: float, *, threshold: float | None = None) -> bool:
    """Classify rainfall value (mm) as rain/dry using configured threshold."""
    if threshold is None:
        threshold = rain_threshold_mm()
    return float(value) > float(threshold)


def rain_probability_threshold(*, config: dict | None = None) -> float:
    """Probability threshold for deciding whether to emit non-zero rainfall prediction."""
    return float(_training_cfg(config).get("rain_probability_threshold", 0.5))
