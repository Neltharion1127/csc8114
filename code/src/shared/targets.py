import math

import torch

from src.shared.common import cfg


def target_transform_mode() -> str:
    """Return the configured target transform mode."""
    return cfg.get("training", {}).get("target_transform", "none")


def transform_target_scalar(value: float) -> float:
    """Map a raw rainfall target into training space."""
    mode = target_transform_mode()
    value = max(float(value), 0.0)
    if mode == "log1p":
        return math.log1p(value)
    return value


def inverse_target_scalar(value: float) -> float:
    """Map a prediction from training space back into raw rainfall units."""
    mode = target_transform_mode()
    value = float(value)
    if mode == "log1p":
        return max(math.expm1(value), 0.0)
    return value


def transform_target_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Map a raw rainfall tensor into training space."""
    mode = target_transform_mode()
    if mode == "log1p":
        return torch.log1p(torch.clamp(tensor, min=0.0))
    return tensor
