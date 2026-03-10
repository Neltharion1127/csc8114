import grpc
from concurrent import futures
import time
import torch
import numpy as np
from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.models.split_lstm import ServerHead
import sys
import os

# Use shared common module (provides `cfg` and `project_root`)
from src.shared.common import cfg, project_root

class FSLServerServicer(fsl_pb2_grpc.FSLServiceServicer):
    """
    Implementation of the Federated Split Learning (FSL) Server logic.
    Responsible for completing the forward pass and calculating loss/gradients.
    """
    def __init__(self):
        # Initialize server-side regressor using module-level cfg
        self.hidden_size = cfg.get("model", {}).get("hidden_size", 64)
        lr = cfg.get("training", {}).get("lr", 0.001)

        self.server_model = ServerHead(hidden_size=self.hidden_size, output_size=1)
        self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        print("[SERVER] ServerHead model initialized and ready for training.")
    
    def Forward(self, request, context):
        """
        Handles incoming smashed activations from distributed clients.
        """
        try:
            # 1. Decode the byte data back into a Torch Tensor
            # Expected shape from client: (batch_size, hidden_size)
            activation_buffer = np.frombuffer(request.activation_data, dtype=np.float32)
            # Reshape according to configured hidden size. Using .clone() ensures memory safety.
            smashed_activation = torch.tensor(activation_buffer, dtype=torch.float32).view(-1, self.hidden_size).detach().clone()
            smashed_activation.requires_grad_(True) # Enable gradient tracking for backprop
            
            target = torch.tensor([[request.true_target]], dtype=torch.float32)
            
            # 2. Complete the forward pass through the ServerHead regressor
            prediction = self.server_model(smashed_activation)
            
            # 3. Calculate Loss (Mean Squared Error) for rainfall prediction
            loss = self.criterion(prediction, target)
            
            # 4. Execute backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # 5. Extract gradients for the smashed activation to send back to the client
            # This allows the client to update the ClientLSTM parameters
            if smashed_activation.grad is not None:
                activation_gradient = smashed_activation.grad.numpy().tobytes()
            else:
                raise ValueError("Gradient calculation failed on the smashed activation.")
            
            # Optional: Update Server-side parameters (ServerHead)
            self.optimizer.step()
            print(f"[SERVER] Node # Client ID:{request.client_id} | Loss: {loss.item():.6f} | Target: {request.true_target}")

            # --- 改動部分：強化日誌監控 ---
            target_val = request.true_target
            pred_val = prediction.item()
            
            if target_val > 0:
                # rain
                print(f"[RAIN EVENT] Client:{request.client_id} | Target:{target_val:.2f} | Pred:{pred_val:.4f} | Loss:{loss.item():.6f}")
            else:
                # no rain
                print(f"[TRAIN] Client:{request.client_id} | Loss:{loss.item():.6f}")

            # 6. Construct the response containing the gradient feedback
            return fsl_pb2.ForwardResponse(
                gradient_data=activation_gradient,
                status_message=f"Success: Loss {loss.item():.4f} calculated."
            )

        except Exception as e:
            print(f"[SERVER ERROR] Processing failed: {str(e)}")
            return fsl_pb2.ForwardResponse(status_message=f"Error: {str(e)}")

def serve():

    # Start a gRPC server with configurable worker threads (cfg loaded at module scope)
    max_workers = cfg.get("server", {}).get("max_workers", 10)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    # Attach our custom logic to the server
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(FSLServerServicer(), server)
    
    # Listen on configured gRPC port (cfg loaded at module scope)
    server_port = cfg.get("grpc", {}).get("server_port", 50051)
    server.add_insecure_port(f'[::]:{server_port}')
    server.start()
    print(f"[SERVER] Listening for incoming FSL connections on port {server_port}...")

    try:
        # Keep the process alive
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
