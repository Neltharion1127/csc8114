import grpc
from concurrent import futures
import time
import torch
import numpy as np
from proto import fsl_pb2
from proto import fsl_pb2_grpc
# Importing the server-side architecture
from models.split_lstm import ServerHead


class FSLServerServicer(fsl_pb2_grpc.FSLServiceServicer):
    """
    Implementation of the Federated Split Learning (FSL) Server logic.
    Responsible for completing the forward pass and calculating loss/gradients.
    """
    def __init__(self):
        # Initialize the server-side regressor (MLP)
        self.server_model = ServerHead(hidden_size=64, output_size=1)
        self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=0.001)
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
            smashed_activation = torch.tensor(activation_buffer, dtype=torch.float32).view(-1, 64)
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
            activation_gradient = smashed_activation.grad.numpy().tobytes()
            
            # Optional: Update Server-side parameters
            self.optimizer.step()

            print(f"[SERVER] Node #{request.client_id} | Loss: {loss.item():.6f} | Target: {request.true_target}")

            # 6. Construct the response containing the gradient feedback
            return fsl_pb2.ForwardResponse(
                gradient_data=activation_gradient,
                status_message=f"Success: Loss {loss.item():.4f} calculated."
            )

        except Exception as e:
            print(f"[SERVER ERROR] Processing failed: {str(e)}")
            return fsl_pb2.ForwardResponse(status_message=f"Error: {str(e)}")

def serve():
    # Start a gRPC server with 10 worker threads
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Attach our custom logic to the server
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(FSLServerServicer(), server)
    
    # Listen on all IP addresses on port 50051
    server.add_insecure_port('[::]:50051')
    server.start()
    
    print("[SERVER] Listening for incoming FSL connections on port 50051...")
    
    # Keep the server alive
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
