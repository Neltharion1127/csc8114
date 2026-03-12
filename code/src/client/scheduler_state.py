from dataclasses import dataclass


@dataclass
class SchedulerState:
    compression_mode: str = "float32"
    last_latency_ms: float = 0.0

    def update(self, log_entry: dict) -> None:
        self.compression_mode = log_entry["NextCompression"]
        self.last_latency_ms = log_entry["LatencyMs"]
