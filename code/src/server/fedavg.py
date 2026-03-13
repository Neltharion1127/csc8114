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

        self.client_weights_buffer: dict[int, dict] = {}
        self.global_weights = None
        self.current_round = 0
        self.lock = threading.Lock()
        self.round_cond = threading.Condition(self.lock)
        self.round_start_time: float | None = None

    def synchronize(self, request, *, local_weights, server_model, optimizer) -> fsl_pb2.SyncResponse:
        with self.round_cond:
            target_round = self.current_round + 1
            now = time.time()
            if not self.client_weights_buffer:
                self.round_start_time = now
            self.client_weights_buffer[request.client_id] = local_weights
            barrier_elapsed_s = (now - self.round_start_time) if self.round_start_time is not None else 0.0
            print(
                f"[FED AVG] Received weights from Client:{request.client_id}. "
                f"Buffer size: {len(self.client_weights_buffer)}/{self.num_clients} | "
                f"barrier_elapsed={barrier_elapsed_s:.2f}s"
            )

            if len(self.client_weights_buffer) >= self.num_clients:
                self._aggregate(
                    server_model=server_model,
                    optimizer=optimizer,
                    barrier_elapsed_s=barrier_elapsed_s,
                )
                self.round_cond.notify_all()
            else:
                remaining = 60.0
                while self.current_round < target_round and remaining > 0:
                    start_wait = time.time()
                    self.round_cond.wait(timeout=remaining)
                    remaining -= time.time() - start_wait

            if self.current_round < target_round or self.global_weights is None:
                # Roll back this client's contribution if the round timed out, so the next
                # synchronization round starts with a clean and correct buffer.
                self.client_weights_buffer.pop(request.client_id, None)
                if not self.client_weights_buffer:
                    self.round_start_time = None
                raise TimeoutError("Timeout waiting for global model aggregation.")
            current_round = self.current_round

        global_weights_bytes = tensor_to_bytes(self.global_weights)
        return fsl_pb2.SyncResponse(
            global_weights=global_weights_bytes,
            round_number=current_round,
        )

    def _aggregate(self, *, server_model, optimizer, barrier_elapsed_s: float) -> None:
        aggregate_start = time.time()
        print(
            f"[FED AVG] Round {self.current_round + 1}: Aggregating {len(self.client_weights_buffer)} models... "
            f"(waited {barrier_elapsed_s:.2f}s for all clients)"
        )

        weights_list = list(self.client_weights_buffer.values())
        self.global_weights = copy.deepcopy(weights_list[0])
        for key in self.global_weights.keys():
            for i in range(1, len(weights_list)):
                self.global_weights[key] += weights_list[i][key]
            self.global_weights[key] = torch.div(self.global_weights[key], len(weights_list))

        self.client_weights_buffer = {}
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

        aggregate_elapsed_s = time.time() - aggregate_start
        self.round_start_time = None
        print(
            f"[FED AVG] Successfully updated global model to Round {self.current_round} "
            f"(aggregate_time={aggregate_elapsed_s:.2f}s)"
        )
        print(f"[SERVER] Best ckpt: {self.session_id}/{Path(server_model_path).name}")

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
        stem_parts = path.stem.split("_")
        try:
            round_number = int(stem_parts[3])
        except (IndexError, ValueError):
            round_number = -1
        return round_number, path.name
