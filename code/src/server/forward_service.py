from dataclasses import dataclass
from datetime import datetime
import time
import threading

import torch

from proto import fsl_pb2
from src.shared.compression import compress, decompress


@dataclass
class ForwardPassResult:
    response: fsl_pb2.ForwardResponse
    log_entry: dict
    monitor_message: str


def handle_forward_request(
    request,
    *,
    hidden_size: int,
    device: torch.device,
    server_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
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

    with sync_lock:
        start_comp_time = time.time()
        prediction = server_model(smashed_activation)
        loss = criterion(prediction, target)

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

        comp_time = (time.time() - start_comp_time) * 1000.0

    target_val = request.true_target
    pred_val = prediction.item()
    loss_val = loss.item()
    current_lr = optimizer.param_groups[0]["lr"]
    rain_correct = int((target_val > 0.1) == (pred_val > 0.1))
    reported_latency = getattr(request, "latency_ms", 0.0)
    payload_bytes = getattr(request, "payload_bytes", 0)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "round": current_round,
        "client_id": request.client_id,
        "is_training": int(is_training),
        "rain_flag": int(target_val > 0.1),
        "rain_correct": rain_correct,
        "compression_mode": compression_mode,
        "next_compression": assigned_compression,
        "profiler_enabled": int(profiler_enabled),
        "scheduler_enabled": int(scheduler_enabled),
        "reported_latency_ms": reported_latency,
        "payload_bytes": payload_bytes,
        "target": target_val,
        "prediction": pred_val,
        "loss": loss_val,
        "learning_rate": current_lr,
        "decompression_time_ms": decomp_time,
        "computation_time_ms": comp_time,
        "gradient_magnitude": grad_mag,
    }

    if target_val > 0:
        monitor_message = f"[💧] ID:{request.client_id} | Tgt:{target_val:.2f} | Loss:{loss_val:.4f}"
    else:
        monitor_message = f"[☁️] ID:{request.client_id} | Loss:{loss_val:.4f}"
    monitor_message = f"{monitor_message} | {compression_mode} [D:{decomp_time:.1f}ms, C:{comp_time:.1f}ms, G:{grad_mag:.3f}]"

    response = fsl_pb2.ForwardResponse(
        gradient_data=activation_gradient,
        status_message=f"Success: Loss {loss_val:.4f} Pred {pred_val:.4f}",
        next_compression_mode=assigned_compression,
        success=True,
        loss=loss_val,
        prediction=pred_val,
    )

    return ForwardPassResult(
        response=response,
        log_entry=log_entry,
        monitor_message=monitor_message,
    )
