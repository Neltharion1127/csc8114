import copy
import os
import time
from datetime import datetime
from pathlib import Path
import threading

import torch

from proto import fsl_pb2
from src.shared.serialization import tensor_to_bytes


class FedAvgCoordinator:
    def __init__(
        self,
        *,
        num_clients: int,
        hidden_size: int,
        session_id: str,
        session_dir: str,
        periodic_dir: str,
        ckpt_interval: int,
    ):
        self.num_clients = num_clients
        self.hidden_size = hidden_size
        self.session_id = session_id
        self.session_dir = session_dir
        self.periodic_dir = periodic_dir
        self.ckpt_interval = ckpt_interval

        self.client_weights_buffer = []
        self.global_weights = None
        self.current_round = 0
        self.lock = threading.Lock()

    def synchronize(self, request, *, local_weights, server_model, optimizer) -> fsl_pb2.SyncResponse:
        with self.lock:
            self.client_weights_buffer.append(local_weights)
            print(
                f"[FED AVG] Received weights from Client:{request.client_id}. "
                f"Buffer size: {len(self.client_weights_buffer)}/{self.num_clients}"
            )

            if len(self.client_weights_buffer) >= self.num_clients:
                self._aggregate(server_model=server_model, optimizer=optimizer)

            current_round = self.current_round

        wait_time = 0
        while self.global_weights is None and wait_time < 60:
            time.sleep(1)
            wait_time += 1

        if self.global_weights is None:
            raise TimeoutError("Timeout waiting for global model aggregation.")

        global_weights_bytes = tensor_to_bytes(self.global_weights)
        return fsl_pb2.SyncResponse(
            global_weights=global_weights_bytes,
            round_number=current_round,
        )

    def _aggregate(self, *, server_model, optimizer) -> None:
        print(f"[FED AVG] Round {self.current_round + 1}: Aggregating {len(self.client_weights_buffer)} models...")

        self.global_weights = copy.deepcopy(self.client_weights_buffer[0])
        for key in self.global_weights.keys():
            for i in range(1, len(self.client_weights_buffer)):
                self.global_weights[key] += self.client_weights_buffer[i][key]
            self.global_weights[key] = torch.div(self.global_weights[key], len(self.client_weights_buffer))

        self.client_weights_buffer = []
        self.current_round += 1

        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        server_ckpt = {
            "round": self.current_round,
            "model_state_dict": server_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "hidden_size": self.hidden_size,
            },
            "session_id": self.session_id,
        }
        server_model_path = os.path.join(
            self.session_dir,
            f"server_head_round_{self.current_round}_{stamp}.pth",
        )
        torch.save(server_ckpt, server_model_path)

        old_checkpoints = sorted(
            Path(self.session_dir).glob("server_head_round_*.pth"),
            key=self._checkpoint_sort_key,
        )
        for old_ckpt in old_checkpoints[:-1]:
            try:
                os.remove(old_ckpt)
            except Exception:
                pass

        if self.current_round % self.ckpt_interval == 0:
            periodic_path = os.path.join(
                self.periodic_dir,
                f"server_round_{self.current_round:04d}.pth",
            )
            torch.save(server_ckpt, periodic_path)
            print(f"[SERVER] Periodic ckpt saved: round {self.current_round:04d}")

        print(f"[FED AVG] Successfully updated global model to Round {self.current_round}")
        print(f"[SERVER] Best ckpt: {self.session_id}/{Path(server_model_path).name}")

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
        stem_parts = path.stem.split("_")
        try:
            round_number = int(stem_parts[3])
        except (IndexError, ValueError):
            round_number = -1
        return round_number, path.name
