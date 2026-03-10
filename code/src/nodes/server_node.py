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
from src.shared.serialization import bytes_to_tensor, tensor_to_bytes
import copy
import threading

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
        
        # FedAvg state & lock
        self.client_weights_buffer = []
        self.global_weights = None
        self.current_round = 0
        self.num_clients = cfg.get("federated", {}).get("num_clients", 3)
        self.sync_lock = threading.Lock() # 確保併發安全
        
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
            smashed_activation = decompress(request.activation_data, (-1, self.hidden_size), compression_mode)
            smashed_activation = smashed_activation.detach().clone().requires_grad_(True)
            decomp_time = (time.time() - start_decomp_time) * 1000.0
            
            target = torch.tensor([[request.true_target]], dtype=torch.float32)
            
            # --- 關鍵改動：使用 Lock 保護共享資源 ---
            with self.sync_lock:
                start_comp_time = time.time()
                # 2.Forward pass
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

                # 準備日誌數值
                target_val = request.true_target
                pred_val = prediction.item()
                loss_val = loss.item()
                
                # 6. 記錄日誌 (避免多線程同時寫入 list)
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

            # --- 輸出監控 (鎖外執行，不影響運算效能) ---
            if target_val > 0:
                # rain
                msg = f"[💧] ID:{request.client_id} | Tgt:{target_val:.2f} | Loss:{loss_val:.4f}"
            else:
                # no rain
                msg = f"[☁️] ID:{request.client_id} | Loss:{loss_val:.4f}"
                
            print(f"{msg} | {compression_mode} [D:{decomp_time:.1f}ms, C:{comp_time:.1f}ms, G:{grad_mag:.3f}]")

        
            # Save periodically (e.g. every 10 requests)
            if len(self.server_logs) % 10 == 0:
                self._save_logs()

            # 6. Construct the response containing the gradient feedback
            return fsl_pb2.ForwardResponse(
                gradient_data=activation_gradient,
                status_message=f"Success: Loss {loss_val:.4f}"
            )

        except Exception as e:
            print(f"[SERVER ERROR] Processing failed: {str(e)}")
            return fsl_pb2.ForwardResponse(status_message=f"Error: {str(e)}")
            
    def _save_logs(self):
        if len(self.server_logs) > 0:
            df = pd.DataFrame(self.server_logs)
            df.to_csv(self.log_file, index=False)
            print(f"[SERVER LOG] Appended {len(self.server_logs)} records to {self.log_file}")
            
    def Synchronize(self, request, context):
        """
        Receives local ClientLSTM weights, aggregats them when enough clients connect,
        and returns the updated Global ClientLSTM weights.
        """
        try:
            # 1. Deserialize the incoming weights
            local_weights = bytes_to_tensor(request.client_weights)
            
            with self.sync_lock:
                self.client_weights_buffer.append(local_weights)
                print(f"[FED AVG] Received weights from Client:{request.client_id}. Buffer size: {len(self.client_weights_buffer)}/{self.num_clients}")
                
                # Check if we have received weights from all clients for this round
                if len(self.client_weights_buffer) >= self.num_clients:
                    print(f"[FED AVG] Round {self.current_round + 1}: Aggregating {len(self.client_weights_buffer)} models...")
                    
                    # 2. Perform Federated Averaging (FedAvg)
                    self.global_weights = copy.deepcopy(self.client_weights_buffer[0])
                    for key in self.global_weights.keys():
                        for i in range(1, len(self.client_weights_buffer)):
                            self.global_weights[key] += self.client_weights_buffer[i][key]
                        self.global_weights[key] = torch.div(self.global_weights[key], len(self.client_weights_buffer))
                    
                    # 3. Reset buffer and increment round
                    self.client_weights_buffer = []
                    self.current_round += 1
                    print(f"[FED AVG] Successfully updated global model to Round {self.current_round}")
            
            # 4. Wait for the global model to be updated if we were the early client
            # (In a real production system, you'd use a more robust waiting/event mechanism)
            wait_time = 0
            while self.global_weights is None and wait_time < 60:
                time.sleep(1)
                wait_time += 1
                
            if self.global_weights is None:
                raise TimeoutError("Timeout waiting for global model aggregation.")
                
            # 5. Serialize and return the aggregated global weights
            global_weights_bytes = tensor_to_bytes(self.global_weights)
            
            return fsl_pb2.SyncResponse(
                global_weights=global_weights_bytes,
                round_number=self.current_round
            )
            
        except Exception as e:
            print(f"[FED AVG ERROR] Synchronization failed: {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return fsl_pb2.SyncResponse()

def serve():

    # Start a gRPC server with configurable worker threads (cfg loaded at module scope)
    max_workers = cfg.get("server", {}).get("max_workers", 10)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    
    servicer = FSLServerServicer()
    # Attach our custom logic to the server
    fsl_pb2_grpc.add_FSLServiceServicer_to_server(servicer, server)
    
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
        if hasattr(servicer, '_save_logs'):
             servicer._save_logs()
                 
        server.stop(0)

if __name__ == '__main__':
    serve()
