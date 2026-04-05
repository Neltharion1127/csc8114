import torch
import torch.nn as nn
from src.models.split_lstm import ClientLSTM, ServerHead

def test_fsl_communication_loop():
    print("Initializing FSL Models...")
    
    # 1. Initialize models
    client_model = ClientLSTM()
    server_model = ServerHead()
    
    # Setup Optimizers
    client_optim = torch.optim.Adam(client_model.parameters(), lr=0.001)
    server_optim = torch.optim.Adam(server_model.parameters(), lr=0.001)
    
    criterion = nn.MSELoss()

    # 2. Setup mock data (batch_size=32, seq_len=24, input_size=5)
    print("Generating mock data from dataloader (32, 24, 5)...")
    mock_x = torch.randn(32, 24, 5) 
    mock_y = torch.randn(32, 1)

    # --- SIMULATE FSL TRAINING STEP ---
    print("\n--- Starting FSL Step ---")

    # [CLIENT SIDE] 
    # Client executes forward pass
    client_optim.zero_grad()
    smashed_activation = client_model(mock_x)
    
    print(f"Client Output Shape (Smashed Activation): {smashed_activation.shape}")
    
    # Client 'sends' this tensor to the server.
    # CRITICAL: We must detach the tensor so the Server doesn't try to backpropagate 
    # directly into the Client's graph (which would fail over a real network).
    # Then we call requires_grad_() so the Server computes the gradient specifically for this intermediate tensor.
    received_activation = smashed_activation.detach().clone()
    received_activation.requires_grad_(True)
    
    # [SERVER SIDE]
    # Server executes forward pass
    # ServerHead returns (rain_logit, rain_amount); use the classifier head for the loss.
    server_optim.zero_grad()
    rain_logit, _ = server_model(received_activation)
    print(f"Server Prediction Shape: {rain_logit.shape}")

    # Server calculates Loss
    loss = criterion(rain_logit, mock_y)
    print(f"Loss computed: {loss.item():.4f}")
    
    # Server executes backward pass
    loss.backward()
    
    # Server updates its own weights
    server_optim.step()
    
    # [NETWORK TRANSFER BACK]
    # Server extracts the gradient at the cut-layer (the input to the ServerHead)
    # This is what gets sent back across the network to the Client!
    gradient_for_client = received_activation.grad.clone()
    print(f"Gradient extracted for Client. Shape: {gradient_for_client.shape}")

    # [CLIENT SIDE]
    # Client receives the gradient and continues the backward pass into its own LSTM weights!
    smashed_activation.backward(gradient_for_client)
    
    # Client updates its own weights
    client_optim.step()
    
    print("\n[SUCCESS] FSL Forward and Backward Pass completed without errors!")

if __name__ == "__main__":
    test_fsl_communication_loop()
