import grpc
import torch
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.models.split_lstm import ClientLSTM
import time
import os
import sys
import json

# Use shared common module (provides `cfg` and `project_root`)
from src.shared.common import cfg, project_root

# Feature columns (can be overridden in config.yaml under data.feature_cols)
FEATURE_COLS = cfg.get("data", {}).get(
    "feature_cols",
    ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"],
)

def run_all_client(data_dir="dataset/processed", epochs=10):
    # Apply configurable defaults (config loaded at module scope)
    time.sleep(cfg.get("training", {}).get("start_delay", 8))
    print("[CLIENT] Connecting to Server....")

    # Locate all sensor files
    search_path = os.path.join(project_root, data_dir, "*.parquet")
    parquet_files = glob.glob(search_path)
    print(f"[CLIENT] Found {len(parquet_files)} sensors. Initializing distributed training...")
    
    # 1. Initialize the edge model (client-side model)
    # Determine input dimension from config or feature list to avoid mismatches
    client_input = cfg.get("model", {}).get("input_size", len(FEATURE_COLS))
    client_hidden = cfg.get("model", {}).get("hidden_size", 64)
    client_model = ClientLSTM(input_size=client_input, hidden_size=client_hidden)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=cfg.get("training", {}).get("lr", 0.001))
    # client_model.eval() #eval 不會訓練, 用在 Testing/Inference Mode, 想要測試模型對「新感測器」的預測準確度時
    client_model.train() # Training Mode

    # Experimental tracking for CSV
    experimental_logs = []

    # 2. Establish gRPC Connection 
    # docker: 'fsl-server:50051'
    # local: 'localhost:50051'
    server_host = cfg.get("grpc", {}).get("server_host", "fsl-server")
    server_port = cfg.get("grpc", {}).get("server_port", 50051)
    target_address = f"{server_host}:{server_port}"
    print(f" [CLIENT] Connecting to server at {target_address}...")
    with grpc.insecure_channel(target_address) as channel:
        stub = fsl_pb2_grpc.FSLServiceStub(channel)
        
        for epoch in range(epochs):  
            print(f"[EPOCH {epoch+1}/{epochs}] Starting distributed training cycle...")
            
            # Iterate through each sensor to simulate distributed edge devices
            for file_path in parquet_files:
                # # Clear gradients for the new sensor
                optimizer.zero_grad()
                sensor_id = Path(file_path).stem
                
                try:
                    # 3. Load and preprocess data
                    df = pd.read_parquet(file_path)
                    feature_cols = FEATURE_COLS
                    raw_data = torch.tensor(df[feature_cols].values[-24:], dtype=torch.float32)
                    input_tensor = raw_data.unsqueeze(0)  # Shape: (1, 24, 5)
                    target_value = float(df['Rain'].iloc[-1])

                    # 4. Forward Pass (Split Point)
                    # This generates 'Smashed Activations' to preserve raw data privacy
                    smashed_activation = client_model(input_tensor)

                    # 5. Prepare and Transmit Data
                    # Detach from graph to convert to bytes for network transfer
                    activation_bytes = smashed_activation.detach().numpy().tobytes()

                    request = fsl_pb2.ForwardRequest(
                        client_id=1, 
                        activation_data=activation_bytes,
                        true_target=target_value,
                        latency_ms=0.0  # Set to 0.0 for ideal network simulation
                    )
                    
                    print(f"[CLIENT] Transmitting activations for {sensor_id}...")

                    # 6. Send activations to Server and receive Gradient Feedback
                    start_time = time.time()
                    response = stub.Forward(request)
                    latency_ms = (time.time() - start_time) * 1000.0

                    # --- NEW: Monitor Non-Zero Rainfall Data ---
                    # Parse Loss from the response string (e.g., "Success: Loss 0.0040")
                    try:
                        current_loss = float(response.status_message.split("Loss")[-1].split()[0].strip())
                    except:
                        current_loss = 0.0

                    # Specialized logging for rain events (Target > 0)
                    if target_value > 0:
                        print(f"[RAIN EVENT] Sensor: {sensor_id[:10]} | Target: {target_value:.2f} | Loss: {current_loss:.6f}")
                    else:
                        print(f"[NO RAIN] Sensor: {sensor_id[:10]} | Loss: {current_loss:.6f}")
                    
                    # 7. Gradient Reconstruction
                    # Convert bytes back to tensor and match the activation shape
                    grad_buffer = np.frombuffer(response.gradient_data, dtype=np.float32)
                    received_grad = torch.from_numpy(grad_buffer).view(smashed_activation.shape)

                    # 8. Backward Pass (Injecting Server Gradients)
                    # This is the core of Split Learning backpropagation
                    smashed_activation.backward(received_grad)

                    # 9. Optimizer Step
                    # Update Client-side Model Weights
                    optimizer.step()
                    print(f"[SERVER] Feedback processed for {sensor_id} | {response.status_message} | Latency: {latency_ms:.2f} ms")

                    # Record data for later analysis
                    experimental_logs.append({
                        "Epoch": epoch + 1,
                        "Sensor": sensor_id,
                        "Target": target_value,
                        "RainFlag": 1 if target_value > 0 else 0,
                        "Loss": current_loss,
                        "LatencyMs": float(latency_ms)
                    })

                except Exception as e:
                    print(f"[CLIENT ERROR] Failed to process {sensor_id}: {str(e)}")

    # --- SAVE RESULTS ---
    output_dir = os.path.join(project_root, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_df = pd.DataFrame(experimental_logs)
    # Simple: write with timestamped filename (YYYYMMDD_HHMMSS)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"training_log_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    log_df.to_csv(filepath, index=False)

    # Save run parameters / config metadata alongside the CSV
    meta = {
        "timestamp": timestamp,
        "csv": filename,
        "num_records": len(experimental_logs),
        "cfg": cfg
    }
    meta_path = filepath.replace('.csv', '_meta.json')
    try:
        with open(meta_path, 'w') as mf:
            json.dump(meta, mf, indent=2)
        print(f"[CLIENT] Saved training log to {filepath}")
        print(f"[CLIENT] Saved run metadata to {meta_path}")
    except Exception as e:
        print(f"[CLIENT WARN] Failed to write metadata: {e}")

    print("\n========== Training complete for all sensors.==========")
if __name__ == '__main__':
    run_all_client()
