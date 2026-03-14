import random
from dataclasses import dataclass

from src.shared.common import cfg


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _offset_for_client(client_id: int, offsets: list[float]) -> float:
    if not offsets:
        return 0.0
    if client_id <= 0:
        return float(offsets[0])
    idx = min(client_id - 1, len(offsets) - 1)
    return float(offsets[idx])


@dataclass
class LatencyConfig:
    base_latency_ms: float
    jitter_ms: float
    burst_every_steps: int
    burst_latency_ms: float
    sleep_fraction: float
    max_sleep_ms: float
    measured_floor_ratio: float
    client_offsets_ms: list[float]


def load_latency_config() -> LatencyConfig:
    profiler_cfg = cfg.get("profiler", {}) if isinstance(cfg, dict) else {}
    latency_cfg = profiler_cfg.get("latency_generator", {})
    if not isinstance(latency_cfg, dict):
        latency_cfg = {}

    offsets_raw = latency_cfg.get("client_offsets_ms", [0.0, 4.0, 9.0])
    if isinstance(offsets_raw, list) and offsets_raw:
        offsets = [_to_float(v, 0.0) for v in offsets_raw]
    else:
        offsets = [0.0, 4.0, 9.0]

    return LatencyConfig(
        base_latency_ms=max(0.0, _to_float(latency_cfg.get("base_latency_ms", 1.5), 1.5)),
        jitter_ms=max(0.0, _to_float(latency_cfg.get("jitter_ms", 0.8), 0.8)),
        burst_every_steps=max(0, _to_int(latency_cfg.get("burst_every_steps", 0), 0)),
        burst_latency_ms=max(0.0, _to_float(latency_cfg.get("burst_latency_ms", 0.0), 0.0)),
        sleep_fraction=max(0.0, _to_float(latency_cfg.get("sleep_fraction", 0.0), 0.0)),
        max_sleep_ms=max(0.0, _to_float(latency_cfg.get("max_sleep_ms", 0.0), 0.0)),
        measured_floor_ratio=max(0.0, _to_float(latency_cfg.get("measured_floor_ratio", 0.25), 0.25)),
        client_offsets_ms=offsets,
    )


class LatencyGenerator:
    """Client-side synthetic latency generator used for scheduler experiments."""

    def __init__(self, *, client_id: int):
        self.client_id = int(client_id)
        self.cfg = load_latency_config()
        base_seed = _to_int(cfg.get("training", {}).get("seed", 42), 42)
        self._rng = random.Random(base_seed + 9973 + self.client_id)
        self._step = 0

    def next_latency_ms(self, *, measured_latency_ms: float) -> float:
        self._step += 1
        offset = _offset_for_client(self.client_id, self.cfg.client_offsets_ms)
        latency = self.cfg.base_latency_ms + offset

        if self.cfg.jitter_ms > 0.0:
            latency += self._rng.gauss(0.0, self.cfg.jitter_ms)

        if self.cfg.burst_every_steps > 0 and (self._step % self.cfg.burst_every_steps) == 0:
            latency += self.cfg.burst_latency_ms

        # Keep synthetic latency from going unrealistically below observed local RTT.
        measured_floor = max(0.0, float(measured_latency_ms)) * self.cfg.measured_floor_ratio
        latency = max(latency, measured_floor, 0.0)
        return float(latency)

    def suggested_sleep_ms(self, *, reported_latency_ms: float) -> float:
        if self.cfg.sleep_fraction <= 0.0:
            return 0.0
        sleep_ms = float(reported_latency_ms) * self.cfg.sleep_fraction
        if self.cfg.max_sleep_ms > 0.0:
            sleep_ms = min(sleep_ms, self.cfg.max_sleep_ms)
        return max(0.0, sleep_ms)
