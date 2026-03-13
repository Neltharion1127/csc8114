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
    best_test_f1: float = float("-inf")
    no_improvement_count: int = 0
    best_model_path: str | None = None


def evaluate_epoch(
    *,
    client_id: int,
    client_model,
    optimizer,
    current_round: int,
    epoch: int,
    avg_val_loss: float,
    val_metrics: dict[str, float | int],
    session_id: str,
    session_dir: str,
    periodic_dir: str,
    patience: int,
    ckpt_interval: int,
    state: CheckpointState,
) -> bool:
    current_f1 = float(val_metrics.get("f1", 0.0))
    current_precision = float(val_metrics.get("precision", 0.0))
    current_recall = float(val_metrics.get("recall", 0.0))
    current_threshold = float(val_metrics.get("selected_threshold", cfg.get("training", {}).get("rain_probability_threshold", 0.5)))
    num_layers_ckpt = sum(
        1 for key in client_model.state_dict()
        if key.startswith("lstm.weight_ih_l")
    )
    base_ckpt = {
        "round": current_round,
        "epoch": epoch + 1,
        "model_state_dict": client_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_val_loss,
        "classification_metrics": {
            "phase": str(val_metrics.get("phase", "VAL")),
            "f1": current_f1,
            "precision": current_precision,
            "recall": current_recall,
            "accuracy": float(val_metrics.get("accuracy", 0.0)),
            "threshold": current_threshold,
            "tp": int(val_metrics.get("tp", 0)),
            "fn": int(val_metrics.get("fn", 0)),
            "fp": int(val_metrics.get("fp", 0)),
            "tn": int(val_metrics.get("tn", 0)),
            "samples": int(
                int(val_metrics.get("tp", 0))
                + int(val_metrics.get("fn", 0))
                + int(val_metrics.get("fp", 0))
                + int(val_metrics.get("tn", 0))
            ),
        },
        "config": {
            "hidden_size": cfg.get("model", {}).get("hidden_size", 64),
            "num_layers": num_layers_ckpt,
            "input_size": cfg.get("model", {}).get("input_size", 5),
        },
        "config_snapshot": copy.deepcopy(cfg),
        "session_id": session_id,
        "client_id": client_id,
    }

    score_improved = current_f1 > state.best_test_f1 + 1e-9
    tie_break_improved = abs(current_f1 - state.best_test_f1) <= 1e-9 and avg_val_loss < state.best_test_loss - 1e-9
    if score_improved or tie_break_improved:
        state.best_test_f1 = current_f1
        state.best_test_loss = avg_val_loss
        state.no_improvement_count = 0
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        state.best_model_path = os.path.join(
            session_dir,
            f"best_client_{client_id}_round_{current_round}_model_{stamp}.pth",
        )
        torch.save(base_ckpt, state.best_model_path)
        print(
            f"[CLIENT {client_id}] New Best! Round {current_round}, "
            f"F1={state.best_test_f1:.4f}, Loss={state.best_test_loss:.4f}, "
            f"Threshold={current_threshold:.3f} -> {session_id}/{Path(state.best_model_path).name}"
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
