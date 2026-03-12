import io

import torch

from proto import fsl_pb2
from src.models.split_lstm import ClientLSTM


def fed_avg_sync(stub, client_id: int, client_model: ClientLSTM) -> ClientLSTM:
    """
    Synchronize local client weights with the server and load the returned global model.
    """
    buffer = io.BytesIO()
    torch.save(client_model.state_dict(), buffer)
    client_weights_bytes = buffer.getvalue()

    sync_req = fsl_pb2.SyncRequest(
        client_id=client_id,
        client_weights=client_weights_bytes,
    )

    print(f"[CLIENT {client_id}] Waiting for global aggregation...")
    sync_res = stub.Synchronize(sync_req)

    if sync_res.global_weights:
        global_buffer = io.BytesIO(sync_res.global_weights)
        global_state_dict = torch.load(global_buffer, weights_only=True, map_location="cpu")
        client_model.load_state_dict(global_state_dict)
        print(f"[CLIENT {client_id}] Successfully loaded Global Model Round {sync_res.round_number}")
    else:
        print(f"[CLIENT {client_id}] Failed to get global weights.")

    return client_model
