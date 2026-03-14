import time

import pandas as pd
import torch

from proto import fsl_pb2
from src.client.latency_generator import LatencyGenerator
from src.models.split_lstm import ClientLSTM
from src.shared.common import cfg
from src.shared.compression import compress, decompress
from src.shared.runtime import maybe_autocast
from src.shared.targets import is_rain, transform_target_scalar

_LATENCY_GENERATORS: dict[int, LatencyGenerator] = {}


def _latency_generator_for(client_id: int) -> LatencyGenerator:
    generator = _LATENCY_GENERATORS.get(int(client_id))
    if generator is None:
        generator = LatencyGenerator(client_id=int(client_id))
        _LATENCY_GENERATORS[int(client_id)] = generator
    return generator


def run_forward_step(
    stub,
    client_id: int,
    client_model: ClientLSTM,
    optimizer: torch.optim.Optimizer,
    df: pd.DataFrame,
    target_idx: int,
    target_value: float,
    mode: str,
    sensor_id: str,
    compression_mode: str,
    feature_cols: list[str],
    feat_stats: tuple | None = None,
    device: torch.device = torch.device("cpu"),
    is_training: bool = True,
    last_latency_ms: float = 0.0,
    seq_len: int | None = None,
) -> dict:
    """
    Runs one split-learning request/response cycle and returns the step log.
    """
    log_step_details = cfg.get("console", {}).get("log_step_details", False)
    profiler_enabled = cfg.get("profiler", {}).get("enabled", True)
    cls_weight = float(cfg.get("training", {}).get("classification_loss_weight", 1.0))
    reg_weight = float(cfg.get("training", {}).get("regression_loss_weight", 1.0))
    seq_len = int(seq_len if seq_len is not None else cfg.get("model", {}).get("seq_len", 24))

    raw_data = df[feature_cols].iloc[target_idx - seq_len:target_idx].values
    if feat_stats:
        mean, std = feat_stats
        raw_data = (raw_data - mean) / std

    raw_data_tensor = torch.tensor(raw_data, dtype=torch.float32, device=device)
    input_tensor = raw_data_tensor.unsqueeze(0)
    with maybe_autocast(device):
        smashed_activation = client_model(input_tensor)

    start_time = time.time()
    activation_bytes = compress(smashed_activation, compression_mode)
    payload_size = len(activation_bytes)
    if profiler_enabled:
        latency_generator = _latency_generator_for(client_id)
        reported_latency_ms = latency_generator.next_latency_ms(
            measured_latency_ms=last_latency_ms,
        )
        sleep_ms = latency_generator.suggested_sleep_ms(
            reported_latency_ms=reported_latency_ms,
        )
        if sleep_ms > 0.0:
            time.sleep(sleep_ms / 1000.0)
    else:
        reported_latency_ms = 0.0
    reported_payload_bytes = payload_size if profiler_enabled else 0
    training_target = transform_target_scalar(target_value)

    request = fsl_pb2.ForwardRequest(
        client_id=client_id,
        activation_data=activation_bytes,
        true_target=training_target,
        latency_ms=reported_latency_ms,
        compression_mode=compression_mode,
        is_training=is_training,
        payload_bytes=reported_payload_bytes,
        raw_target=target_value,
        classification_loss_weight=cls_weight,
        regression_loss_weight=reg_weight,
    )
    phase = "TRAIN" if is_training else "TEST"
    if log_step_details:
        print(f"[{phase}] Transmitting activations for {sensor_id}... Payload: {payload_size} bytes")

    response = stub.Forward(request)
    latency_ms = (time.time() - start_time) * 1000.0 if profiler_enabled else 0.0

    if not response.success:
        raise RuntimeError(response.status_message or "Server forward pass failed.")

    current_loss = float(response.loss)
    prediction_val = float(response.prediction)
    rain_probability = float(getattr(response, "rain_probability", 0.0))
    classification_loss = float(getattr(response, "classification_loss", 0.0))
    regression_loss = float(getattr(response, "regression_loss", 0.0))

    if log_step_details:
        icon = "RAIN" if is_rain(target_value) else "DRY"
        print(f"[{icon}] [{mode}] {sensor_id[:10]} | 3h Target: {target_value:.2f} | Loss: {current_loss:.6f}")

    if is_training:
        received_grad = decompress(response.gradient_data, smashed_activation.shape, compression_mode).to(device)
        smashed_activation.backward(received_grad)
        torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=1.0)
        optimizer.step()

    if log_step_details:
        print(f"[SERVER] Feedback processed | {response.status_message} | Latency: {latency_ms:.2f} ms")
    scheduler_enabled = cfg.get("scheduler", {}).get("enabled", True)
    next_compression_mode = compression_mode
    next_rho = int(cfg.get("federated", {}).get("rho", 1))
    if scheduler_enabled and response.next_compression_mode:
        next_compression_mode = response.next_compression_mode
    if scheduler_enabled and getattr(response, "next_rho", 0) > 0:
        next_rho = int(response.next_rho)

    return {
        "Target": target_value,
        "Prediction": prediction_val,
        "RainFlag": int(is_rain(target_value)),
        "Loss": current_loss,
        "RainProbability": rain_probability,
        "ClassificationLoss": classification_loss,
        "RegressionLoss": regression_loss,
        "LatencyMs": float(latency_ms),
        "PayloadBytes": reported_payload_bytes,
        "NextCompression": next_compression_mode,
        "NextRho": int(next_rho),
        "ProfilerEnabled": int(profiler_enabled),
        "SchedulerEnabled": int(scheduler_enabled),
    }
