from dataclasses import dataclass


@dataclass
class SchedulerState:
    compression_mode: str = "float32"
    rho: int = 1
    last_latency_ms: float = 0.0

    def update(self, log_entry: dict) -> None:
        self.compression_mode = log_entry["NextCompression"]
        self.rho = max(1, int(log_entry.get("NextRho", self.rho)))
        self.last_latency_ms = log_entry["LatencyMs"]
