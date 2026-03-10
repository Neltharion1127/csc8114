import torch
import numpy as np

def compress(tensor: torch.Tensor, mode: str) -> bytes:
    """
    Compresses a PyTorch tensor into bytes based on the specified mode.
    Modes: 'float32', 'float16', 'int8'
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
        
    else:  # float32 or default
        return arr.astype(np.float32).tobytes()

def decompress(data_bytes: bytes, shape: tuple, mode: str) -> torch.Tensor:
    """
    Decompresses bytes back into a PyTorch tensor.
    """
    if mode == "float16":
        arr = np.frombuffer(data_bytes, dtype=np.float16).astype(np.float32)
        
    elif mode == "int8":
        # Extract the scale (first 4 bytes)
        scale = np.frombuffer(data_bytes[:4], dtype=np.float32)[0]
        # Dequantize the rest
        quantized = np.frombuffer(data_bytes[4:], dtype=np.int8).astype(np.float32)
        arr = quantized * scale
        
    else:  # float32 or default
        arr = np.frombuffer(data_bytes, dtype=np.float32)
        
    return torch.from_numpy(arr).view(shape)
