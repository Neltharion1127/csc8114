import io
import os
import time
from dataclasses import dataclass

import torch

from proto import fsl_pb2
from src.models.split_lstm import ClientLSTM


@dataclass
class SyncResult:
    client_model: ClientLSTM
    round_number: int
    accepted: bool
    applied_round: int
    refresh_only: bool
    status_message: str
    sync_bytes_sent: int = 0   # bytes sent (client weights upload)
    sync_bytes_recv: int = 0   # bytes received (global weights download)


def fed_avg_sync(
    stub,
    client_id: int,
    client_model: ClientLSTM,
    *,
    model_round: int,
    local_epochs: int,
) -> SyncResult:
    """
    Synchronize local client weights with the server and load the returned global model.
    """
    buffer = io.BytesIO()
    torch.save(client_model.state_dict(), buffer)
    client_weights_bytes = buffer.getvalue()

    sync_req = fsl_pb2.SyncRequest(
        client_id=client_id,
        client_weights=client_weights_bytes,
        base_round=int(model_round),
        local_epochs=int(local_epochs),
    )

    print(
        f"[CLIENT {client_id}] Waiting for global aggregation... "
        f"(base_round={int(model_round)} local_epochs={int(local_epochs)})"
    )
    wait_start = time.time()
    sync_res = stub.Synchronize(sync_req, metadata=[("scenario-id", os.environ.get("SCENARIO_ID", ""))])
    wait_elapsed_s = time.time() - wait_start

    round_number = int(getattr(sync_res, "round_number", model_round))
    accepted = bool(getattr(sync_res, "accepted", False))
    applied_round = int(getattr(sync_res, "applied_round", 0))
    refresh_only = bool(getattr(sync_res, "refresh_only", False))
    status_message = str(getattr(sync_res, "status_message", "")).strip()

    if sync_res.global_weights:
        global_buffer = io.BytesIO(sync_res.global_weights)
        global_state_dict = torch.load(global_buffer, weights_only=True, map_location="cpu")
        client_model.load_state_dict(global_state_dict)

    if accepted:
        print(
            f"[CLIENT {client_id}] Global model updated to Round {round_number} "
            f"(applied_round={applied_round} sync_wait={wait_elapsed_s:.2f}s)"
        )
    elif refresh_only:
        print(
            f"[CLIENT {client_id}] Refreshed local model to Round {round_number} "
            f"without contributing this update (sync_wait={wait_elapsed_s:.2f}s)"
        )
    else:
        print(
            f"[CLIENT {client_id}] Synchronization completed without model update "
            f"(sync_wait={wait_elapsed_s:.2f}s)"
        )

    if status_message:
        print(f"[CLIENT {client_id}] Sync status: {status_message}")

    sync_bytes_sent = len(client_weights_bytes)
    sync_bytes_recv = len(sync_res.global_weights) if sync_res.global_weights else 0

    return SyncResult(
        client_model=client_model,
        round_number=round_number,
        accepted=accepted,
        applied_round=applied_round,
        refresh_only=refresh_only,
        status_message=status_message,
        sync_bytes_sent=sync_bytes_sent,
        sync_bytes_recv=sync_bytes_recv,
    )
