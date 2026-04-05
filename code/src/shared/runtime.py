import os
import random
from contextlib import nullcontext

import numpy as np
import torch

from src.shared.common import cfg


def resolve_device() -> torch.device:
    """
    Resolve runtime device in a cross-platform way.

    Priority:
    1) FSL_DEVICE env
    2) training.device in config
    3) legacy training.use_gpu flag
    """
    training_cfg = cfg.get("training", {})
    requested = os.getenv("FSL_DEVICE", "").strip().lower()
    if not requested:
        requested = str(training_cfg.get("device", "auto")).strip().lower()

    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[RUNTIME] CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[RUNTIME] MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    legacy_use_gpu = bool(training_cfg.get("use_gpu", False))
    if legacy_use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def grpc_channel_options() -> list[tuple[str, int]]:
    """Build gRPC channel options: message size limits + keepalive for real-network stability."""
    max_mb = int(cfg.get("grpc", {}).get("max_message_mb", 50))
    max_bytes = max_mb * 1024 * 1024
    return [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
        # Send a ping every 30 s even when there are no active RPCs, so both sides
        # detect a dead TCP connection before the next training step hits it.
        ("grpc.keepalive_time_ms", 30_000),
        # If the peer does not respond to the ping within 10 s, declare the link dead.
        ("grpc.keepalive_timeout_ms", 10_000),
        # Allow pings even when no RPC is in flight (required for the above to work).
        ("grpc.keepalive_permit_without_calls", True),
        ("grpc.http2.max_pings_without_data", 0),
    ]


def create_grpc_channel(address: str):
    """
    Create a gRPC channel to *address* with keepalive options.

    When ``grpc.tls_enabled`` is true in config, an SSL/TLS channel is returned
    and the server certificate (or CA bundle) is read from ``grpc.tls_cert_path``.
    Otherwise an insecure channel is returned (safe when running over Tailscale).
    """
    import grpc as _grpc  # local import to avoid circular dependency at module level

    options = grpc_channel_options()
    grpc_cfg = cfg.get("grpc", {})

    if grpc_cfg.get("tls_enabled", False):
        ca_cert_path = grpc_cfg.get("tls_cert_path")
        if ca_cert_path:
            with open(ca_cert_path, "rb") as fh:
                root_certificates = fh.read()
        else:
            root_certificates = None  # fall back to system CA bundle
        credentials = _grpc.ssl_channel_credentials(root_certificates=root_certificates)
        return _grpc.secure_channel(address, credentials, options=options)

    return _grpc.insecure_channel(address, options=options)


def resolve_server_address() -> str:
    """Resolve server address with env override support."""
    grpc_cfg = cfg.get("grpc", {})
    host = os.getenv("FSL_SERVER_HOST", str(grpc_cfg.get("server_host", "fsl-server")))
    port = int(os.getenv("FSL_SERVER_PORT", str(grpc_cfg.get("server_port", 50051))))
    return f"{host}:{port}"


def maybe_autocast(device: torch.device):
    """
    Optional AMP autocast context.
    Enabled when training.mixed_precision is set to 'auto' or 'bf16'.
    """
    mixed = str(cfg.get("training", {}).get("mixed_precision", "none")).lower().strip()
    if mixed == "none":
        return nullcontext()

    if mixed in {"auto", "bf16"}:
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)

    return nullcontext()


def set_global_seed(seed: int | None, *, role: str = "runtime") -> int | None:
    """
    Set Python/NumPy/PyTorch random seeds for reproducibility.
    Returns the applied seed, or None if seeding is disabled.
    """
    if seed is None:
        print(f"[RUNTIME] Seed disabled for {role}.")
        return None

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[RUNTIME] Seed set for {role}: {seed}")
    return seed
