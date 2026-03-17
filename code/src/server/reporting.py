import os
import threading

import pandas as pd

from src.shared.common import cfg, project_root


class ServerReporter:
    def __init__(self, *, session_id: str):
        self.server_logs = []
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self.total_records = 0
        self.log_dir = os.path.join(project_root, "results", session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"server_log_{session_id}.csv")
        self.flush_interval = max(1, int(cfg.get("server", {}).get("log_flush_interval", 100)))
        print(f"[SERVER] Server log path: {self.log_file}")
        print(f"[SERVER] Server log flush interval: {self.flush_interval} records")

    def record(self, log_entry: dict) -> None:
        batch = None
        with self._lock:
            self.server_logs.append(log_entry)
            if len(self.server_logs) >= self.flush_interval:
                batch = self.server_logs
                self.server_logs = []
        if batch:
            self._flush_batch(batch)

    def flush(self) -> None:
        batch = None
        with self._lock:
            if self.server_logs:
                batch = self.server_logs
                self.server_logs = []
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[dict]) -> None:
        if not batch:
            return

        # Serialize file writes to avoid duplicate CSV headers under concurrent flushes.
        with self._flush_lock:
            try:
                df = pd.DataFrame(batch)
                file_exists = os.path.exists(self.log_file)
                df.to_csv(
                    self.log_file,
                    mode="a",
                    header=not file_exists,
                    index=False,
                )
                self.total_records += len(batch)
                print(
                    f"[SERVER LOG] Appended {len(batch)} records (total={self.total_records}) "
                    f"to {self.log_file}"
                )
            except Exception:
                # Keep records for retry on next flush if a transient write error occurs.
                with self._lock:
                    self.server_logs = batch + self.server_logs
                raise

