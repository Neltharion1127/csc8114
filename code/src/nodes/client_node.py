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
from src.shared.compression import compress, decompress

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
    
    # 1. Initialize the edge model
    client_input = cfg.get("model", {}).get("input_size", len(FEATURE_COLS))
    client_hidden = cfg.get("model", {}).get("hidden_size", 64)
    client_model = ClientLSTM(input_size=client_input, hidden_size=client_hidden)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=cfg.get("training", {}).get("lr", 0.001))
    # client_model.eval() #eval 不會訓練, 用在 Testing/Inference Mode, 想要測試模型對「新感測器」的預測準確度時
    client_model.train() # Training Mode

    # 建議 2026/03/01 當作測試預測目標，之前的用來訓練
    split_date = pd.Timestamp("2026-02-28 00:00:00")

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
                    
                    # --- 核心修改：計算未來 3 小時降雨總量 ---
                    # 使用 shift(-3) 取得未來數據，滾動窗口 3 小時加總
                    df['future_3h_rain'] = df['Rain'].shift(-3).rolling(window=3).sum()
                    
                    split_pos = df.index.get_indexer([split_date], method='pad')[0]
                    all_indices = np.arange(len(df))
                    
                    # 建立過濾條件：時間在 split_date 之前，且符合歷史/未來窗口邊界
                    # 限制：24 <= index < len(df) - 3 確保資料完整性
                    base_mask = (all_indices < split_pos) & (all_indices >= 24) & (all_indices < len(df) - 3)

                    rainy_pos = all_indices[base_mask & (df['future_3h_rain'] > 0)]
                    dry_pos = all_indices[base_mask & (df['future_3h_rain'] == 0)]

                    # --- 平衡採樣 (Balanced Sampling) ---
                    # 50% 機率強迫學習「有雨」樣本，解決數據不平衡問題
                    if len(rainy_pos) > 0 and np.random.rand() > 0.5:
                        target_idx = np.random.choice(rainy_pos)
                        mode = "RAIN_SAMPLE"
                    elif len(dry_pos) > 0:
                        target_idx = np.random.choice(dry_pos)
                        mode = "DRY_SAMPLE"
                    else:
                        continue # 資料不足則跳過此感測器

                    # 提取特徵：過去 24 小時歷史 (24, 5)
                    raw_data = torch.tensor(df[FEATURE_COLS].iloc[target_idx-24:target_idx].values, dtype=torch.float32)
                    input_tensor = raw_data.unsqueeze(0) # 增加 Batch 維度 -> (1, 24, 5)
                    
                    # 提取目標：未來 3 小時累積降雨量
                    target_value = float(df['future_3h_rain'].iloc[target_idx])

                    # 4. Forward Pass
                    smashed_activation = client_model(input_tensor)
                    
                    # --- Compression ---
                    compression_mode = cfg.get("compression", {}).get("mode", "float32")
                    start_time = time.time()
                    
                    activation_bytes = compress(smashed_activation, compression_mode)
                    payload_size = len(activation_bytes)

                    request = fsl_pb2.ForwardRequest(
                        client_id=1, 
                        activation_data=activation_bytes,
                        true_target=target_value,
                        latency_ms=0.0,  # Set to 0.0 for ideal network simulation
                        compression_mode=compression_mode
                    )                  
                    print(f"[CLIENT] Transmitting activations for {sensor_id}... Payload: {payload_size} bytes")
                    print(f"[CLIENT] Transmitting activations for {sensor_id}...")

                    # 6. Send activations to Server and receive Gradient Feedback
                    response = stub.Forward(request)
                    latency_ms = (time.time() - start_time) * 1000.0

                    # --- NEW: Monitor Non-Zero Rainfall Data ---
                    # Parse Loss from the response string (e.g., "Success: Loss 0.0040")
                    try:
                        current_loss = float(response.status_message.split("Loss")[-1].split()[0].strip())
                    except:
                        current_loss = 0.0

                    # 終端機顯示日誌
                    icon = "💧💧💧" if target_value > 0 else "☁️"
                    print(f"{icon} [{mode}] Sensor: {sensor_id[:10]} | 3h Target: {target_value:.2f} | Loss: {current_loss:.6f}")
                    
                    # 7. Gradient Reconstruction & Backward Pass
                    # expected shape: smashed_activation.shape -> (1, 64) etc.
                    received_grad = decompress(response.gradient_data, smashed_activation.shape, compression_mode)
                    smashed_activation.backward(received_grad)

                    # 8. Optimizer Step
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
                        "LatencyMs": float(latency_ms),
                        "PayloadBytes": payload_size
                    })

                except Exception as e:
                    print(f"[CLIENT ERROR] Failed to process {sensor_id}: {str(e)}")

    # --- SAVE RESULTS ---
    output_dir = os.path.join(project_root, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    log_df = pd.DataFrame(experimental_logs)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"training_log_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    log_df.to_csv(filepath, index=False)
    print(f"[CLIENT] Saved training log to {filepath}")

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
        print(f"[CLIENT] Saved metadata to {meta_path}")
    except Exception as e:
        print(f"[CLIENT WARN] Failed to write metadata: {e}")

    print("\n========== Training complete for all sensors.==========")
if __name__ == '__main__':
    run_all_client()
