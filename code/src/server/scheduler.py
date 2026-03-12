import threading


class CompressionScheduler:
    def __init__(
        self,
        default_mode: str = "float32",
        *,
        enabled: bool = True,
        float16_threshold: float = 4.0,
        int8_threshold: float = 10.0,
    ):
        self.default_mode = default_mode
        self.enabled = enabled
        self.float16_threshold = float16_threshold
        self.int8_threshold = int8_threshold
        self._client_modes: dict[int, str] = {}
        self._lock = threading.Lock()

    def assign(self, client_id: int, reported_latency: float) -> str:
        with self._lock:
            if client_id not in self._client_modes:
                self._client_modes[client_id] = self.default_mode

            if not self.enabled:
                self._client_modes[client_id] = self.default_mode
            elif reported_latency > 0:
                if reported_latency > self.int8_threshold:
                    self._client_modes[client_id] = "int8"
                elif reported_latency > self.float16_threshold:
                    self._client_modes[client_id] = "float16"
                else:
                    self._client_modes[client_id] = self.default_mode

            return self._client_modes[client_id]
