import copy
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import threading
from typing import Any

import torch

from proto import fsl_pb2
from src.shared.config_artifacts import build_config_ref, build_config_snapshot
from src.shared.serialization import tensor_to_bytes
from src.shared.common import cfg


@dataclass
class PendingUpdate:
    client_id: int
    weights: dict[str, torch.Tensor]
    base_round: int
    local_epochs: int
    arrived_at: float


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
        min_clients_per_round: int = 2,
        round_timeout_sec: float = 15.0,
        grace_period_sec: float = 0.0,
    ):
        self.num_clients = num_clients
        self.hidden_size = hidden_size
        self.session_id = session_id
        self.session_dir = session_dir
        self.periodic_dir = periodic_dir
        self.ckpt_interval = ckpt_interval
        self.min_clients_per_round = max(1, int(min_clients_per_round))
        self.round_timeout_sec = max(0.1, float(round_timeout_sec))
        self.grace_period_sec = max(0.0, float(grace_period_sec))
        self._quorum_reached_at: float | None = None
        self._startup_deadline: float | None = None

        self.client_weights_buffer: dict[int, PendingUpdate] = {}
        self.global_weights = None
        self.current_round = 0
        self.lock = threading.Lock()
        self.round_cond = threading.Condition(self.lock)
        self.round_start_time: float | None = None
        self.round_error: str | None = None
        self._expected_schema: dict[str, tuple[tuple[int, ...], torch.dtype]] | None = None
        self._active_clients: set[int] = set()
        self._completed_clients: set[int] = set()

    @staticmethod
    def _validate_weights_object(local_weights: Any) -> dict[str, torch.Tensor]:
        if not isinstance(local_weights, dict) or not local_weights:
            raise ValueError("Client weights must be a non-empty state_dict-like dict.")
        normalized: dict[str, torch.Tensor] = {}
        for key, value in local_weights.items():
            if not isinstance(key, str):
                raise ValueError("State dict keys must be strings.")
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"State dict value for key '{key}' is not a torch.Tensor.")
            normalized[key] = value
        return normalized

    @staticmethod
    def _build_schema(weights: dict[str, torch.Tensor]) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        return {key: (tuple(tensor.shape), tensor.dtype) for key, tensor in weights.items()}

    def _validate_against_schema(self, weights: dict[str, torch.Tensor]) -> None:
        schema = self._expected_schema
        if schema is None:
            self._expected_schema = self._build_schema(weights)
            return

        current_keys = set(weights.keys())
        expected_keys = set(schema.keys())
        if current_keys != expected_keys:
            missing = sorted(expected_keys - current_keys)
            extra = sorted(current_keys - expected_keys)
            raise ValueError(
                f"State dict key mismatch. missing={missing[:5]} extra={extra[:5]}"
            )

        for key, tensor in weights.items():
            expected_shape, expected_dtype = schema[key]
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Tensor shape mismatch for '{key}': got {tuple(tensor.shape)} expected {expected_shape}"
                )
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Tensor dtype mismatch for '{key}': got {tensor.dtype} expected {expected_dtype}"
                )

    def register_client(self, client_id: int) -> None:
        with self.round_cond:
            if client_id not in self._completed_clients:
                self._active_clients.add(int(client_id))
            self.round_cond.notify_all()

    def mark_client_completed(self, client_id: int, *, server_model, optimizer) -> None:
        with self.round_cond:
            client_id = int(client_id)
            self._completed_clients.add(client_id)
            self._active_clients.discard(client_id)
            print(
                f"[FED AVG] Client {client_id} marked complete. "
                f"active_clients={len(self._active_clients)}"
            )
            if self.client_weights_buffer and self._has_quorum_locked():
                self._aggregate_locked(
                    server_model=server_model,
                    optimizer=optimizer,
                    reason="active-set-updated",
                )
            self.round_cond.notify_all()

    def synchronize(self, request, *, local_weights, server_model, optimizer) -> fsl_pb2.SyncResponse:
        client_id = int(request.client_id)
        base_round = int(getattr(request, "base_round", self.current_round))
        local_epochs = int(getattr(request, "local_epochs", 0))

        with self.round_cond:
            if client_id not in self._completed_clients:
                self._active_clients.add(client_id)

            # --- Safety Catch: Prevent rounds exceeding config ---
            num_rounds = cfg.get("training", {}).get("num_rounds", 30)
            if self.current_round >= num_rounds:
                print(f"[FED AVG] Target rounds ({num_rounds}) reached. Rejecting update from Client:{client_id}")
                return self._build_sync_response_locked(
                    accepted=False,
                    applied_round=self.current_round,
                    status_message="FINISHED",
                    refresh_only=True
                )

            if base_round < self.current_round:
                print(
                    f"[FED AVG] Rejecting stale update from Client:{client_id} "
                    f"(base_round={base_round}, current_round={self.current_round})"
                )
                return self._build_sync_response_locked(
                    accepted=False,
                    applied_round=0,
                    refresh_only=True,
                    status_message=(
                        f"Stale update rejected: client base_round={base_round}, "
                        f"server current_round={self.current_round}."
                    ),
                )

            if base_round > self.current_round:
                print(
                    f"[FED AVG] Client:{client_id} is ahead of server state "
                    f"(base_round={base_round}, current_round={self.current_round})"
                )
                return self._build_sync_response_locked(
                    accepted=False,
                    applied_round=0,
                    refresh_only=self.global_weights is not None,
                    status_message=(
                        f"Client is ahead of server state: client base_round={base_round}, "
                        f"server current_round={self.current_round}."
                    ),
                )

            # Before the first round, wait until enough clients have connected.
            # Deadline is measured from when the FIRST client enters this wait,
            # not from server startup — the server may have been up for a long time
            # before clients are deployed via Ansible.
            if self.current_round == 0:
                if self._startup_deadline is None:
                    self._startup_deadline = time.time() + self.round_timeout_sec
                    print(
                        f"[FED AVG] Startup wait begun: waiting up to {self.round_timeout_sec:.0f}s "
                        f"for {self.min_clients_per_round} clients "
                        f"(currently {len(self._active_clients)} registered)."
                    )
                while len(self._active_clients) < self.min_clients_per_round:
                    remaining = self._startup_deadline - time.time()
                    if remaining <= 0:
                        print(
                            f"[FED AVG] Startup wait timed out: only "
                            f"{len(self._active_clients)}/{self.min_clients_per_round} "
                            f"clients connected. Proceeding with current active set."
                        )
                        break
                    self.round_cond.wait(timeout=min(remaining, 5.0))

            target_round = self.current_round + 1
            now = time.time()
            if not self.client_weights_buffer:
                self.round_start_time = now
                self.round_error = None

            validated_weights = self._validate_weights_object(local_weights)
            self._validate_against_schema(validated_weights)
            self.client_weights_buffer[client_id] = PendingUpdate(
                client_id=client_id,
                weights=validated_weights,
                base_round=base_round,
                local_epochs=local_epochs,
                arrived_at=now,
            )
            barrier_elapsed_s = (now - self.round_start_time) if self.round_start_time is not None else 0.0
            required = self._required_clients_locked()
            print(
                f"[FED AVG] Received weights from Client:{client_id}. "
                f"Buffer size: {len(self.client_weights_buffer)}/{required} "
                f"| active_clients={len(self._active_clients)} "
                f"| base_round={base_round} local_epochs={local_epochs} "
                f"| barrier_elapsed={barrier_elapsed_s:.2f}s"
            )

            self._maybe_aggregate_locked(
                server_model=server_model,
                optimizer=optimizer,
                reason="quorum-reached",
            )

            while self.current_round < target_round and not self.round_error:
                remaining = self._remaining_window_locked()
                if remaining <= 0:
                    if self.client_weights_buffer and self.current_round < target_round:
                        self._aggregate_locked(
                            server_model=server_model,
                            optimizer=optimizer,
                            reason="timeout",
                        )
                    break
                self.round_cond.wait(timeout=remaining)
                if self.current_round < target_round and not self.round_error:
                    self._maybe_aggregate_locked(
                        server_model=server_model,
                        optimizer=optimizer,
                        reason="quorum-reached",
                    )

            if self.round_error:
                raise RuntimeError(self.round_error)

            if self.current_round < target_round or self.global_weights is None:
                raise TimeoutError("Timeout waiting for global model aggregation.")

            return self._build_sync_response_locked(
                accepted=True,
                applied_round=self.current_round,
                refresh_only=False,
                status_message=(
                    f"Accepted into aggregation round {self.current_round} "
                    f"(active_clients={len(self._active_clients)})."
                ),
            )

    def _active_client_count_locked(self) -> int:
        return len(self._active_clients)

    def _required_clients_locked(self) -> int:
        active_count = self._active_client_count_locked()
        if active_count <= 0:
            return 1
        return max(1, min(self.min_clients_per_round, active_count))

    def _remaining_window_locked(self) -> float:
        if self.round_start_time is None:
            return self.round_timeout_sec
        elapsed = time.time() - self.round_start_time
        timeout_remaining = max(0.0, self.round_timeout_sec - elapsed)
        # If quorum is reached and grace period is active, wake up sooner to check it.
        if self._quorum_reached_at is not None and self.grace_period_sec > 0.0:
            grace_remaining = max(0.0, self.grace_period_sec - (time.time() - self._quorum_reached_at))
            return min(timeout_remaining, grace_remaining)
        return timeout_remaining

    def _has_quorum_locked(self) -> bool:
        return len(self.client_weights_buffer) >= self._required_clients_locked()

    def _grace_period_elapsed_locked(self) -> bool:
        """True if grace period has passed since quorum was first reached."""
        if self.grace_period_sec <= 0.0:
            return True
        if self._quorum_reached_at is None:
            return False
        return (time.time() - self._quorum_reached_at) >= self.grace_period_sec

    def _maybe_aggregate_locked(self, *, server_model, optimizer, reason: str) -> None:
        if not (self.client_weights_buffer and self._has_quorum_locked()):
            return
        if self._quorum_reached_at is None:
            self._quorum_reached_at = time.time()
            if self.grace_period_sec > 0.0:
                print(
                    f"[FED AVG] Quorum reached ({len(self.client_weights_buffer)}/"
                    f"{self._required_clients_locked()}), "
                    f"waiting grace period {self.grace_period_sec:.0f}s for stragglers..."
                )
        if self._grace_period_elapsed_locked():
            self._aggregate_locked(server_model=server_model, optimizer=optimizer, reason=reason)

    def _build_sync_response_locked(
        self,
        *,
        accepted: bool,
        applied_round: int,
        refresh_only: bool,
        status_message: str,
    ) -> fsl_pb2.SyncResponse:
        global_weights_bytes = tensor_to_bytes(self.global_weights) if self.global_weights is not None else b""
        return fsl_pb2.SyncResponse(
            global_weights=global_weights_bytes,
            round_number=int(self.current_round),
            accepted=bool(accepted),
            applied_round=int(applied_round),
            refresh_only=bool(refresh_only),
            status_message=status_message,
        )

    def _aggregate_locked(self, *, server_model, optimizer, reason: str) -> None:
        aggregate_start = time.time()
        pending_updates = list(self.client_weights_buffer.values())
        client_ids = [update.client_id for update in pending_updates]
        barrier_elapsed_s = (time.time() - self.round_start_time) if self.round_start_time is not None else 0.0
        print(
            f"[FED AVG] Round {self.current_round + 1}: Aggregating {len(pending_updates)} models "
            f"from clients={client_ids} (active={len(self._active_clients)} "
            f"quorum={self._required_clients_locked()} reason={reason} "
            f"waited={barrier_elapsed_s:.2f}s)"
        )

        total_epochs = sum(u.local_epochs for u in pending_updates)
        if total_epochs <= 0:
            total_epochs = len(pending_updates)
        self.global_weights = copy.deepcopy(pending_updates[0].weights)
        for key in self.global_weights.keys():
            self.global_weights[key] = sum(
                u.weights[key] * (u.local_epochs / total_epochs)
                for u in pending_updates
            )

        self.client_weights_buffer = {}
        self._quorum_reached_at = None
        self.current_round += 1

        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        config_snapshot, snapshot_policy = build_config_snapshot()
        server_ckpt = {
            "round": self.current_round,
            "model_state_dict": server_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "hidden_size": self.hidden_size,
            },
            "config_snapshot_policy": snapshot_policy,
            "config_ref": build_config_ref(),
            "session_id": self.session_id,
            "aggregated_client_ids": client_ids,
            "aggregation_reason": reason,
        }
        if config_snapshot is not None:
            server_ckpt["config_snapshot"] = config_snapshot
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
        self.round_error = None
        print(
            f"[FED AVG] Successfully updated global model to Round {self.current_round} "
            f"(aggregate_time={aggregate_elapsed_s:.2f}s)"
        )
        print(f"[SERVER] Best ckpt: {self.session_id}/{Path(server_model_path).name}")
        self.round_cond.notify_all()

    @staticmethod
    def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
        stem_parts = path.stem.split("_")
        try:
            round_number = int(stem_parts[3])
        except (IndexError, ValueError):
            round_number = -1
        return round_number, path.name
