import os

import pandas as pd

from src.shared.common import project_root


class ServerReporter:
    def __init__(self, *, session_id: str):
        self.server_logs = []
        self.log_dir = os.path.join(project_root, "results", session_id)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"server_log_{session_id}.csv")
        print(f"[SERVER] Server log path: {self.log_file}")

    def record(self, log_entry: dict) -> None:
        self.server_logs.append(log_entry)
        if len(self.server_logs) % 10 == 0:
            self.flush()

    def flush(self) -> None:
        if len(self.server_logs) > 0:
            df = pd.DataFrame(self.server_logs)
            df.to_csv(self.log_file, index=False)
            print(f"[SERVER LOG] Appended {len(self.server_logs)} records to {self.log_file}")
