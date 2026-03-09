import io
import torch

def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Serializes a PyTorch tensor into a raw byte sequence so it can be 
    transmitted over a gRPC network connection.
    
    Args:
        tensor (torch.Tensor): The tensor to serialize (e.g., smashed activation or gradient).
        
    Returns:
        bytes: A compact byte string representation of the tensor.
    """
    buffer = io.BytesIO()
    # torch.save natively handles serialization to file-like objects
    torch.save(tensor, buffer)
    # Extract the raw binary data
    return buffer.getvalue()

def bytes_to_tensor(data: bytes) -> torch.Tensor:
    """
    Deserializes a byte sequence received from the network back into 
    a workable PyTorch tensor. Automatically maps the tensor to CPU initially.
    
    Args:
        data (bytes): The raw byte string received from gRPC.
        
    Returns:
        torch.Tensor: The reconstructed PyTorch tensor.
    """
    buffer = io.BytesIO(data)
    # Ensure it's loaded to CPU first to avoid device-map issues across nodes
    tensor = torch.load(buffer, weights_only=True, map_location='cpu')
    return tensor

if __name__ == "__main__":
    # --- Quick Sanity Check ---
    print("Testing Tensor Serialization Engine...")
    
    # 1. Create a mock tensor locally (similar to a batch of 32 smashed activations)
    original_tensor = torch.randn(32, 64)
    print(f"Original Tensor Shape: {original_tensor.shape}")
    
    # 2. Serialize to bytes (Mocking the process of packing it into the proto message)
    encoded_bytes = tensor_to_bytes(original_tensor)
    print(f"Serialized Byte Length: {len(encoded_bytes)} bytes")
    
    # 3. Deserialize back to tensor (Mocking the receiving end)
    reconstructed_tensor = bytes_to_tensor(encoded_bytes)
    
    # 4. Verify mathematical fidelity
    is_identical = torch.allclose(original_tensor, reconstructed_tensor)
    print(f"Reconstruction 100% Identical: {is_identical}")
