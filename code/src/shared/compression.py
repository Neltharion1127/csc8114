import torch
import numpy as np


def _resolve_topk_ratio() -> float:
    # Local import avoids introducing a hard dependency at module import time.
    from src.shared.common import cfg

    raw = cfg.get("compression", {}).get("topk_ratio", 0.5)
    try:
        ratio = float(raw)
    except (TypeError, ValueError):
        ratio = 0.5
    return float(np.clip(ratio, 1e-6, 1.0))


def compress(tensor: torch.Tensor, mode: str) -> bytes:
    """
    Compresses a PyTorch tensor into bytes based on the specified mode.
    Modes: 'float32', 'float16', 'int8', 'topk'
    """
    arr = tensor.detach().cpu().numpy()
    
    if mode == "float16":
        return arr.astype(np.float16).tobytes()
        
    elif mode == "int8":
        max_abs = np.max(np.abs(arr))
        scale = float(max_abs / 127.0) if max_abs > 0 else 1.0
        quantized = np.round(arr / scale).astype(np.int8)
        # Prepend the scale (4 bytes float32) to the quantized data
        scale_bytes = np.array([scale], dtype=np.float32).tobytes()
        return scale_bytes + quantized.tobytes()
    elif mode == "topk":
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        n = int(flat.size)
        if n == 0:
            # Header only: n=0, k=0
            return np.array([0, 0], dtype=np.int32).tobytes()

        ratio = _resolve_topk_ratio()
        k = int(max(1, min(n, round(n * ratio))))
        if k >= n:
            indices = np.arange(n, dtype=np.int32)
            values = flat
        else:
            # Select entries with largest magnitudes.
            indices = np.argpartition(np.abs(flat), -k)[-k:].astype(np.int32, copy=False)
            values = flat[indices]

        header = np.array([n, int(indices.size)], dtype=np.int32).tobytes()
        return header + indices.tobytes() + values.astype(np.float32, copy=False).tobytes()
        
    else:  # float32 or default
        return arr.astype(np.float32).tobytes()

def decompress(data_bytes: bytes, shape: tuple, mode: str) -> torch.Tensor:
    """
    Decompresses bytes back into a PyTorch tensor.
    """
    if mode == "float16":
        arr = np.frombuffer(data_bytes, dtype=np.float16).astype(np.float32, copy=True)
        
    elif mode == "int8":
        # Extract the scale (first 4 bytes)
        scale = np.frombuffer(data_bytes[:4], dtype=np.float32)[0]
        # Dequantize the rest
        quantized = np.frombuffer(data_bytes[4:], dtype=np.int8).astype(np.float32, copy=True)
        arr = quantized * scale
    elif mode == "topk":
        if len(data_bytes) < 8:
            raise ValueError("Invalid topk payload: header is incomplete.")
        n, k = np.frombuffer(data_bytes[:8], dtype=np.int32)
        n = int(n)
        k = int(k)
        if n < 0 or k < 0:
            raise ValueError(f"Invalid topk payload header: n={n}, k={k}")
        expected = 8 + 4 * k + 4 * k
        if len(data_bytes) != expected:
            raise ValueError(
                f"Invalid topk payload length: expected {expected} bytes, got {len(data_bytes)} bytes."
            )
        dense = np.zeros(n, dtype=np.float32)
        if k > 0:
            indices = np.frombuffer(data_bytes[8 : 8 + 4 * k], dtype=np.int32)
            values = np.frombuffer(data_bytes[8 + 4 * k :], dtype=np.float32)
            dense[indices] = values
        arr = dense
        
    else:  # float32 or default
        arr = np.frombuffer(data_bytes, dtype=np.float32).copy()
        
    return torch.from_numpy(arr).view(shape)
