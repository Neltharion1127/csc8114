import copy
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

import torch

from src.shared.common import cfg


@dataclass
class CheckpointState:
    best_test_loss: float = float("inf")
    no_improvement_count: int = 0
    best_model_path: str | None = None


def evaluate_epoch(
    *,
    client_id: int,
    client_model,
    optimizer,
    current_round: int,
    epoch: int,
    avg_test_loss: float,
    session_id: str,
    session_dir: str,
    periodic_dir: str,
    patience: int,
    ckpt_interval: int,
    state: CheckpointState,
) -> bool:
    num_layers_ckpt = sum(
        1 for key in client_model.state_dict()
        if key.startswith("lstm.weight_ih_l")
    )
    base_ckpt = {
        "round": current_round,
        "epoch": epoch + 1,
        "model_state_dict": client_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_test_loss,
        "config": {
            "hidden_size": cfg.get("model", {}).get("hidden_size", 64),
            "num_layers": num_layers_ckpt,
            "input_size": cfg.get("model", {}).get("input_size", 5),
        },
        "config_snapshot": copy.deepcopy(cfg),
        "session_id": session_id,
        "client_id": client_id,
    }

    if avg_test_loss < state.best_test_loss:
        state.best_test_loss = avg_test_loss
        state.no_improvement_count = 0
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        state.best_model_path = os.path.join(
            session_dir,
            f"best_client_{client_id}_round_{current_round}_model_{stamp}.pth",
        )
        torch.save(base_ckpt, state.best_model_path)
        print(
            f"[CLIENT {client_id}] New Best! Round {current_round}, "
            f"Loss={state.best_test_loss:.4f} -> {session_id}/{Path(state.best_model_path).name}"
        )
    else:
        state.no_improvement_count += 1
        print(f"[CLIENT {client_id}] No improvement for {state.no_improvement_count}/{patience} rounds.")

    if state.no_improvement_count >= patience:
        print(f"\n[EARLY STOP] Client {client_id} triggered at round {current_round} (Patience={patience})")
        return True

    if current_round > 0 and current_round % ckpt_interval == 0:
        periodic_path = os.path.join(
            periodic_dir,
            f"client_{client_id}_round_{current_round:04d}.pth",
        )
        torch.save(base_ckpt, periodic_path)
        print(f"[CLIENT {client_id}] Periodic ckpt saved: round {current_round:04d}")

    return False
