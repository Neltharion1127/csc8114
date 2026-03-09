import grpc
import torch
import pandas as pd
import glob
from pathlib import Path
# Import the auto-generated gRPC code
from proto import fsl_pb2
from proto import fsl_pb2_grpc
from src.models.split_lstm import ClientLSTM
import time

def run_all_client(data_dir="dataset/processed"):
    print("[CLIENT] Waiting 3 seconds for Server to boot up...")
    time.sleep(2)
    print("[CLIENT] Connecting to Server....")

    # 1. Automatically locate all sensor data files (Total 12 sensors expected)
    parquet_files = glob.glob(f"{data_dir}/*.parquet")
    print(f"[CLIENT] Found {len(parquet_files)} sensors. Initializing distributed training...")
    
    # 2. Initialize the edge model (Client side of the split architecture)
    # Input dimension: 5 (Temp, Humidity, Pressure, Wind Speed, Rain)
    client_model = ClientLSTM(input_size=5, hidden_size=64)
    client_model.eval()

    # 3. Establish connection with the Docker Compose hostname "fsl-server"  
    # docker: 'fsl-server:50051'
    # local: 'localhost:50051'
    with grpc.insecure_channel('fsl-server:50051') as channel:
        
        # Create a stub (the local representative of the remote server)
        stub = fsl_pb2_grpc.FSLServiceStub(channel)
        
        # 4. Loop through each sensor to simulate distributed edge devices
        for file_path in parquet_files:
            sensor_id = Path(file_path).stem
            print(f"[CLIENT] Processing node: {sensor_id}")

            try:
                # Load the local dataset for the current sensor
                df = pd.read_parquet(file_path)
                
                # Prepare the 24-hour time series window as input
                feature_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Rain']
                raw_data = torch.tensor(df[feature_cols].values[-24:], dtype=torch.float32)
                input_tensor = raw_data.unsqueeze(0)  # Shape: (1, 24, 5)
                
                # Label: Rainfall amount for the next 1-hour prediction
                target_value = float(df['Rain'].iloc[-1])

                # 5. Execute Forward Pass to generate 'Smashed Activations'
                # Ensures raw edge data remains on-device for privacy enhancement
                with torch.no_grad():
                    smashed_activation = client_model(input_tensor)

                # 6. Transmit activations to Server via gRPC
                request = fsl_pb2.ForwardRequest(
                    client_id=1, 
                    activation_data=smashed_activation.numpy().tobytes(),
                    true_target=target_value,
                    latency_ms=15.2
                )
                
                print(f"[CLIENT] Transmitting activations for {sensor_id}...")
                response = stub.Forward(request)
                
                # 7. Process the feedback (Gradients) returned from the Server
                print(f"[SERVER RESPONSE] Status: {response.status_message}")

 
            except grpc.RpcError as e:
                print(f"[CLIENT ERROR] Failed to process {sensor_id}: {str(e)}")

if __name__ == '__main__':
    run_all_client()
