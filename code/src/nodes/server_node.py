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
import pandas as pd
from datetime import datetime

# Use shared common module (provides `cfg` and `project_root`)
from src.shared.common import cfg, project_root
from src.shared.compression import compress, decompress

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
        
        # Diagnostics logging
        self.server_logs = []
        self.log_dir = os.path.join(project_root, "results")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"server_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

        print("[SERVER] ServerHead model initialized and ready for training.")
    
    def Forward(self, request, context):
        """
        Handles incoming smashed activations from distributed clients.
        """
        try:
            # 1. Decode the byte data back into a Torch Tensor
            # Expected shape from client: (batch_size, hidden_size)
            compression_mode = request.compression_mode if hasattr(request, "compression_mode") and request.compression_mode else "float32"
            
            start_decomp_time = time.time()
            if compression_mode == "float16" or compression_mode == "int8" or compression_mode == "float32":
                smashed_activation = decompress(request.activation_data, (-1, self.hidden_size), compression_mode)
            else:
                smashed_activation = decompress(request.activation_data, (-1, self.hidden_size), "float32")
                
            smashed_activation = smashed_activation.detach().clone()
            smashed_activation.requires_grad_(True) # Enable gradient tracking for backprop
            decomp_time = (time.time() - start_decomp_time) * 1000.0
            
            start_comp_time = time.time()
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
                grad_mag = torch.norm(smashed_activation.grad).item()
                grad_arr = smashed_activation.grad.numpy()
                if compression_mode == "float16":
                    activation_gradient = grad_arr.astype(np.float16).tobytes()
                elif compression_mode == "int8":
                    max_abs = np.max(np.abs(grad_arr))
                    scale = max_abs / 127.0 if max_abs > 0 else 1.0
                    quantized = np.round(grad_arr / scale).astype(np.int8)
                    activation_gradient = np.array([scale], dtype=np.float32).tobytes() + quantized.tobytes()
                else:
                    activation_gradient = grad_arr.tobytes()
            else:
                grad_mag = 0.0
                raise ValueError("Gradient calculation failed on the smashed activation.")
            
            # Optional: Update Server-side parameters (ServerHead)
            self.optimizer.step()
            
            comp_time = (time.time() - start_comp_time) * 1000.0
            
            # --- 改動部分：強化日誌監控 ---
            target_val = request.true_target
            pred_val = prediction.item()
            loss_val = loss.item()
            
            if target_val > 0:
                # rain
                print(f"[RAIN EVENT] Client:{request.client_id} | Target:{target_val:.2f} | Pred:{pred_val:.4f} | Loss:{loss_val:.6f}")
            else:
                # no rain
                print(f"[TRAIN] Client:{request.client_id} | Loss:{loss_val:.6f}")
                
            print(f"      -> Metrics: Decompress {decomp_time:.2f}ms | Compute {comp_time:.2f}ms | Grad Mag {grad_mag:.4f}")

            # Collect log
            self.server_logs.append({
                "timestamp": datetime.now().isoformat(),
                "client_id": request.client_id,
                "compression_mode": compression_mode,
                "target": target_val,
                "prediction": pred_val,
                "loss": loss_val,
                "decompression_time_ms": decomp_time,
                "computation_time_ms": comp_time,
                "gradient_magnitude": grad_mag
            })

            # Save periodically (e.g. every 100 requests)
            if len(self.server_logs) % 100 == 0:
                self._save_logs()

            # 6. Construct the response containing the gradient feedback
            return fsl_pb2.ForwardResponse(
                gradient_data=activation_gradient,
                status_message=f"Success: Loss {loss_val:.4f} calculated."
            )

        except Exception as e:
            print(f"[SERVER ERROR] Processing failed: {str(e)}")
            return fsl_pb2.ForwardResponse(status_message=f"Error: {str(e)}")
            
    def _save_logs(self):
        if len(self.server_logs) > 0:
            df = pd.DataFrame(self.server_logs)
            df.to_csv(self.log_file, index=False)
            print(f"[SERVER LOG] Appended {len(self.server_logs)} records to {self.log_file}")

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
        # Save any remaining logs on shutdown
        for servicer in [s for s in server._state.generic_handlers if isinstance(s, fsl_pb2_grpc.FSLServiceServicer)]:
             if hasattr(servicer, '_save_logs'):
                 servicer._save_logs()
                 
        server.stop(0)

if __name__ == '__main__':
    serve()
