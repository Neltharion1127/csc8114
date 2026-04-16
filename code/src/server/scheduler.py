import threading


class CompressionScheduler:
    def __init__(
        self,
        default_mode: str = "float32",
        *,
        enabled: bool = True,
        float16_threshold: float = 4.0,
        int8_threshold: float = 10.0,
        base_rho: int = 1,
        min_rho: int = 1,
        max_rho: int = 20,
        rho_step: int = 1,
        topk_multiplier: float = 1.5,
        latency_ema_alpha: float = 0.2,
    ):
        self.default_mode = default_mode
        self.enabled = enabled
        self.float16_threshold = float16_threshold
        self.int8_threshold = int8_threshold
        self.base_rho = max(1, int(base_rho))
        self.min_rho = max(1, int(min_rho))
        self.max_rho = max(self.min_rho, int(max_rho))
        self.rho_step = max(0, int(rho_step))
        self.topk_threshold = float(int8_threshold) * float(topk_multiplier)
        self.latency_ema_alpha = float(latency_ema_alpha)
        self._client_state: dict[int, dict[str, float | int | str]] = {}
        self._lock = threading.Lock()

    def assign(self, client_id: int, reported_latency: float) -> tuple[str, int]:
        with self._lock:
            if client_id not in self._client_state:
                self._client_state[client_id] = {
                    "mode": self.default_mode,
                    "rho": self.base_rho,
                    "latency_ema": 0.0,
                }
            state = self._client_state[client_id]

            if not self.enabled:
                state["mode"] = self.default_mode
                state["rho"] = self.base_rho
            elif reported_latency > 0:
                prev_ema = float(state["latency_ema"])
                alpha = min(max(self.latency_ema_alpha, 0.0), 1.0)
                if prev_ema <= 0.0:
                    latency_ema = float(reported_latency)
                else:
                    latency_ema = alpha * float(reported_latency) + (1.0 - alpha) * prev_ema
                state["latency_ema"] = latency_ema

                # Escalate compression level and synchronization interval with network pressure.
                if latency_ema > self.topk_threshold:
                    severity = 3
                    mode = "topk"
                elif latency_ema > self.int8_threshold:
                    severity = 2
                    mode = "int8"
                elif latency_ema > self.float16_threshold:
                    severity = 1
                    mode = "float16"
                else:
                    severity = 0
                    mode = self.default_mode

                rho = self.base_rho + severity * self.rho_step
                state["mode"] = mode
                state["rho"] = max(self.min_rho, min(self.max_rho, int(rho)))

            return str(state["mode"]), int(state["rho"])
