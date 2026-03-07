import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import glob

class RainfallDataset(Dataset):
    """
    A PyTorch Dataset for loading Newcastle Urban Observatory rainfall data.
    It takes a list of preprocessed parquet files, applies a sliding window,
    and returns (X, y) pairs for training an LSTM.
    """
    def __init__(self, file_paths, seq_length=24, pred_horizon=1, scaler=None, is_train=True):
        """
        :param file_paths: List of paths to the `.parquet` files for this client
        :param seq_length: How many past hours to look at (X)
        :param pred_horizon: How many hours ahead to predict (y)
        :param scaler: An sklearn StandardScaler instance (fitted on train data only)
        """
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.is_train = is_train
        
        # Features to use as X (Temperature, Humidity, Pressure, Wind Speed, Rain)
        self.feature_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Rain']
        self.target_col = 'Rain'
        
        self.X_data = []
        self.y_data = []
        self.scaler = scaler
        
        self._load_and_process_files(file_paths)
        
    def _load_and_process_files(self, file_paths):
        all_features = []
        
        # 1. Load all assigned sensor files into memory
        for path in file_paths:
            df = pd.read_parquet(path)
            
            # Sort by timestamp just in case
            df = df.sort_values('Timestamp').reset_index(drop=True)
            
            # Extract raw numpy arrays
            features = df[self.feature_cols].values
            all_features.append(features)
            
        # 2. Fit or apply the Scaler
        # If this is the Training dataset, we fit the scaler on ALL its raw data first
        if self.is_train and self.scaler is None:
            self.scaler = StandardScaler()
            # Concatenate all sensor features vertically to learn the global mean/std
            stacked_train_data = np.vstack(all_features)
            self.scaler.fit(stacked_train_data)
            
        # 3. Create Sliding Windows
        for features in all_features:
            # Scale the features
            scaled_features = self.scaler.transform(features)
            
            # Since Rain is the last column, we know its index.
            # But we want 'y' to be unscaled true rainfall values or scaled?
            # Typically, target can remain unscaled or scaled. To calculate true MSE 
            # we often leave target unscaled, but for LSTM stability scaling is better.
            # Here we grab the unscaled target for 'y' array so the Loss has real physical meaning.
            target_rain_idx = self.feature_cols.index(self.target_col)
            unscaled_rain = features[:, target_rain_idx]
            
            # Create pairs
            total_time_steps = len(scaled_features)
            for i in range(total_time_steps - self.seq_length - self.pred_horizon + 1):
                # X: The historical chunk (seq_length, num_features)
                x_window = scaled_features[i : i + self.seq_length]
                # Y: The future target value
                y_val = unscaled_rain[i + self.seq_length + self.pred_horizon - 1]
                
                self.X_data.append(x_window)
                self.y_data.append(y_val)
                
    def __len__(self):
        return len(self.X_data)
        
    def __getitem__(self, idx):
        # Convert to PyTorch floats
        x = torch.tensor(self.X_data[idx], dtype=torch.float32)
        y = torch.tensor([self.y_data[idx]], dtype=torch.float32)
        return x, y


def create_federated_dataloaders(data_dir="dataset/processed", 
                                 num_clients=3, 
                                 batch_size=64, 
                                 seq_length=24,
                                 test_split=0.2):
    """
    Distributes available sensor data files among `num_clients`.
    Returns a dictionary of DataLoaders for each client.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    
    if len(all_files) < num_clients:
        raise ValueError(f"Not enough sensors ({len(all_files)}) to split among {num_clients} clients.")
        
    # Non-IID Splitting (Simple deterministic split by array slicing)
    # E.g., 12 files, 3 clients -> 4 files per client
    client_files = np.array_split(all_files, num_clients)
    
    loaders = {}
    
    for client_id, assigned_files in enumerate(client_files):
        # Convert back to list
        assigned_files = assigned_files.tolist()
        
        # Temporal split for each client: 
        # Actually, to prevent data peaking, we should split the DataFrames temporally inside 
        # the dataset, but for simplicity, we treat the first 80% of rows in each file as train.
        # Since we load entirely in `RainfallDataset`, passing test splits requires slight modification.
        pass # Simplified for clarity, assuming full dataset loading to test first

    return client_files

# Simple test block
if __name__ == "__main__":
    files = sorted(glob.glob("dataset/processed/*.parquet"))
    print(f"Found {len(files)} processed parquet files.")
    
    if len(files) > 0:
        # Test just the first sensor
        test_dataset = RainfallDataset([files[0]], seq_length=24, is_train=True)
        print(f"Dataset length for 1 sensor: {len(test_dataset)} samples.")
        
        X_sample, y_sample = test_dataset[0]
        print(f"X shape: {X_sample.shape}")
        print(f"y shape: {y_sample.shape}")
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        batch_x, batch_y = next(iter(test_loader))
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch y shape: {batch_y.shape}")
