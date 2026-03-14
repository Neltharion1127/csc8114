import io
import threading

import torch

from proto import fsl_pb2
from src.server.fedavg import FedAvgCoordinator


def _client_state(fill_value: float) -> dict[str, torch.Tensor]:
    return {
        "lstm.weight_ih_l0": torch.full((2, 2), fill_value=fill_value),
        "lstm.weight_hh_l0": torch.full((2, 2), fill_value=fill_value),
    }


def _decode_state_dict(payload: bytes) -> dict[str, torch.Tensor]:
    return torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")


def test_partial_participation_and_stale_refresh(tmp_path):
    session_dir = tmp_path / "session"
    periodic_dir = session_dir / "periodic"
    session_dir.mkdir(parents=True)
    periodic_dir.mkdir(parents=True)

    server_model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
    coordinator = FedAvgCoordinator(
        num_clients=3,
        hidden_size=64,
        session_id="test-session",
        session_dir=str(session_dir),
        periodic_dir=str(periodic_dir),
        ckpt_interval=10,
        min_clients_per_round=2,
        round_timeout_sec=5.0,
    )
    for client_id in (1, 2, 3):
        coordinator.register_client(client_id)

    responses: dict[int, fsl_pb2.SyncResponse] = {}

    def do_sync(client_id: int, fill_value: float) -> None:
        request = fsl_pb2.SyncRequest(
            client_id=client_id,
            client_weights=b"placeholder",
            base_round=0,
            local_epochs=1,
        )
        responses[client_id] = coordinator.synchronize(
            request,
            local_weights=_client_state(fill_value),
            server_model=server_model,
            optimizer=optimizer,
        )

    t1 = threading.Thread(target=do_sync, args=(1, 1.0), daemon=True)
    t1.start()

    do_sync(2, 3.0)
    t1.join(timeout=2.0)
    assert not t1.is_alive()

    res1 = responses[1]
    res2 = responses[2]
    assert res1.accepted is True
    assert res2.accepted is True
    assert res1.round_number == 1
    assert res2.round_number == 1

    aggregated = _decode_state_dict(res1.global_weights)
    assert torch.allclose(aggregated["lstm.weight_ih_l0"], torch.full((2, 2), 2.0))
    assert torch.allclose(aggregated["lstm.weight_hh_l0"], torch.full((2, 2), 2.0))

    stale_response = coordinator.synchronize(
        fsl_pb2.SyncRequest(
            client_id=3,
            client_weights=b"placeholder",
            base_round=0,
            local_epochs=3,
        ),
        local_weights=_client_state(5.0),
        server_model=server_model,
        optimizer=optimizer,
    )
    assert stale_response.accepted is False
    assert stale_response.refresh_only is True
    assert stale_response.round_number == 1

    refreshed = _decode_state_dict(stale_response.global_weights)
    assert torch.allclose(refreshed["lstm.weight_ih_l0"], torch.full((2, 2), 2.0))
