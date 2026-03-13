from dataclasses import dataclass
from datetime import datetime
import time
import threading

import torch
import torch.nn.functional as F

from proto import fsl_pb2
from src.shared.compression import compress, decompress
from src.shared.common import cfg
from src.shared.runtime import maybe_autocast
from src.shared.targets import (
    inverse_target_scalar,
    is_rain,
    rain_probability_threshold,
    rain_threshold_mm,
)


@dataclass
class ForwardPassResult:
    response: fsl_pb2.ForwardResponse
    log_entry: dict
    monitor_message: str


def _classification_loss(
    rain_logit: torch.Tensor,
    rain_target: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    training_cfg = cfg.get("training", {})
    loss_type = str(training_cfg.get("classification_loss_type", "weighted_bce")).strip().lower()
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    focal_alpha = float(training_cfg.get("focal_alpha", -1.0))

    bce_loss = F.binary_cross_entropy_with_logits(
        rain_logit,
        rain_target,
        pos_weight=pos_weight,
        reduction="none",
    )
    if loss_type != "focal":
        return bce_loss.mean()

    prob = torch.sigmoid(rain_logit)
    pt = torch.where(rain_target > 0.5, prob, 1.0 - prob)
    focal_factor = torch.pow(1.0 - pt, focal_gamma)
    loss = focal_factor * bce_loss

    if 0.0 <= focal_alpha <= 1.0:
        alpha_t = rain_target * focal_alpha + (1.0 - rain_target) * (1.0 - focal_alpha)
        loss = alpha_t * loss

    return loss.mean()


def handle_forward_request(
    request,
    *,
    hidden_size: int,
    device: torch.device,
    server_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sync_lock: threading.Lock,
    current_round: int,
    assigned_compression: str,
    profiler_enabled: bool,
    scheduler_enabled: bool,
) -> ForwardPassResult:
    compression_mode = request.compression_mode if hasattr(request, "compression_mode") and request.compression_mode else "float32"
    start_decomp_time = time.time()
    smashed_activation = decompress(request.activation_data, (-1, hidden_size), compression_mode).to(device)
    smashed_activation = smashed_activation.detach().clone().requires_grad_(True)
    decomp_time = (time.time() - start_decomp_time) * 1000.0

    target = torch.tensor([[request.true_target]], dtype=torch.float32, device=device)
    is_training = getattr(request, "is_training", True)
    raw_target_val = getattr(request, "raw_target", request.true_target)
    rain_threshold = rain_threshold_mm()
    prob_threshold = rain_probability_threshold()
    rain_target = torch.tensor([[1.0 if is_rain(raw_target_val, threshold=rain_threshold) else 0.0]], dtype=torch.float32, device=device)
    cls_weight = float(request.classification_loss_weight) if hasattr(request, "classification_loss_weight") and request.classification_loss_weight > 0 else 1.0
    reg_weight = float(request.regression_loss_weight) if hasattr(request, "regression_loss_weight") and request.regression_loss_weight > 0 else 1.0
    pos_weight_value = float(cfg.get("training", {}).get("classification_positive_weight", 1.0))
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    with sync_lock:
        start_comp_time = time.time()
        previous_mode = server_model.training
        server_model.train(mode=is_training)
        try:
            with maybe_autocast(device):
                rain_logit, rain_amount = server_model(smashed_activation)
                cls_loss = _classification_loss(
                    rain_logit,
                    rain_target,
                    pos_weight=pos_weight,
                )
                if rain_target.item() > 0.5:
                    reg_loss = F.smooth_l1_loss(rain_amount, target)
                else:
                    reg_loss = torch.zeros((), dtype=torch.float32, device=device)
                loss = cls_weight * cls_loss + reg_weight * reg_loss

            if is_training:
                optimizer.zero_grad()
                loss.backward()

                if smashed_activation.grad is None:
                    raise ValueError("Gradient calculation failed on the smashed activation.")

                grad_mag = torch.norm(smashed_activation.grad).item()
                activation_gradient = compress(smashed_activation.grad, compression_mode)
                optimizer.step()
            else:
                grad_mag = 0.0
                activation_gradient = b""
        finally:
            server_model.train(mode=previous_mode)

        comp_time = (time.time() - start_comp_time) * 1000.0

    rain_prob = torch.sigmoid(rain_logit).item()
    pred_val = rain_amount.item()
    raw_pred_val = inverse_target_scalar(pred_val) if rain_prob >= prob_threshold else 0.0
    loss_val = loss.item()
    cls_loss_val = cls_loss.item()
    reg_loss_val = reg_loss.item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss)
    current_lr = optimizer.param_groups[0]["lr"]
    rain_correct = int(
        is_rain(raw_target_val, threshold=rain_threshold)
        == is_rain(raw_pred_val, threshold=rain_threshold)
    )
    reported_latency = getattr(request, "latency_ms", 0.0)
    payload_bytes = getattr(request, "payload_bytes", 0)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "round": current_round,
        "client_id": request.client_id,
        "is_training": int(is_training),
        "rain_flag": int(is_rain(raw_target_val, threshold=rain_threshold)),
        "rain_correct": rain_correct,
        "compression_mode": compression_mode,
        "next_compression": assigned_compression,
        "profiler_enabled": int(profiler_enabled),
        "scheduler_enabled": int(scheduler_enabled),
        "reported_latency_ms": reported_latency,
        "payload_bytes": payload_bytes,
        "target": raw_target_val,
        "prediction": raw_pred_val,
        "target_transformed": request.true_target,
        "prediction_transformed": pred_val,
        "rain_probability": rain_prob,
        "loss": loss_val,
        "classification_loss": cls_loss_val,
        "regression_loss": reg_loss_val,
        "learning_rate": current_lr,
        "decompression_time_ms": decomp_time,
        "computation_time_ms": comp_time,
        "gradient_magnitude": grad_mag,
    }

    if raw_target_val > 0:
        monitor_message = (
            f"[💧] ID:{request.client_id} | Tgt:{raw_target_val:.2f} | "
            f"Pred:{raw_pred_val:.2f} | P(rain):{rain_prob:.2f} | Loss:{loss_val:.4f}"
        )
    else:
        monitor_message = (
            f"[☁️] ID:{request.client_id} | Pred:{raw_pred_val:.2f} | "
            f"P(rain):{rain_prob:.2f} | Loss:{loss_val:.4f}"
        )
    monitor_message = f"{monitor_message} | {compression_mode} [D:{decomp_time:.1f}ms, C:{comp_time:.1f}ms, G:{grad_mag:.3f}]"

    response = fsl_pb2.ForwardResponse(
        gradient_data=activation_gradient,
        status_message=f"Success: Loss {loss_val:.4f} Pred {raw_pred_val:.4f} P(rain) {rain_prob:.4f}",
        next_compression_mode=assigned_compression,
        success=True,
        loss=loss_val,
        prediction=raw_pred_val,
        rain_probability=rain_prob,
        classification_loss=cls_loss_val,
        regression_loss=reg_loss_val,
    )

    return ForwardPassResult(
        response=response,
        log_entry=log_entry,
        monitor_message=monitor_message,
    )
