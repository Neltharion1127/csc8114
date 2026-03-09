import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob

# Try importing tqdm for a progress bar, fallback if not installed
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

def preprocess_sensor_data(raw_csv_path: str, output_dir: str):
    """
    Reads the raw UO CSV, pivots variables into columns, resamples to hourly, 
    and saves individual parquet files per sensor.
    """
    raw_path = Path(raw_csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_path.exists():
        print(f"Error: {raw_csv_path} does not exist.")
        return

    print(f"Loading raw data from {raw_csv_path}...")
    # Read the data. We use low_memory=False because the file is huge
    df = pd.read_csv(raw_csv_path, low_memory=False)
    
    # 1. Parsing & Filtering
    print("Parsing timestamps...")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Optional: Filter date range strictly between 2025-03-07 and 2026-03-07
    # start_date = pd.to_datetime("2025-03-07")
    # end_date = pd.to_datetime("2026-03-07")
    # df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    
    # Get unique sensors
    sensors = df['Sensor_Name'].unique()
    print(f"Found {len(sensors)} unique sensors. Processing individually...")
    
    processed_count = 0
    
    # Process each sensor sequentially to avoid memory spikes
    for sensor in tqdm(sensors):
        sensor_df = df[df['Sensor_Name'] == sensor].copy()
        
        # 2. Pivot & Resample
        # Pivot the long format into wide format
        wide_df = sensor_df.pivot_table(
            index='Timestamp', 
            columns='Variable', 
            values='Value',
            aggfunc='mean' # Default aggregation, we will fix Rain later
        )
        
        # Ensure all target columns exist even if this sensor never recorded some
        expected_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Rain']
        for col in expected_cols:
            if col not in wide_df.columns:
                wide_df[col] = np.nan
                
        # Handle the special case for Rain: if it existed, we need to recalculate 
        # its aggregation as sum() instead of mean(). 
        # Actually, since pivot_table already aggregated by mean for identical timestamps, 
        # doing accurate sum requires grouping before pivot. A simpler way for urban observatory data 
        # (where rain is often tipping bucket accumulations) is resampling straight away.
        
        # Resample to 1-hour intervals
        # Weather variables use mean, Rain uses sum
        agg_rules = {
            'Temperature': 'mean',
            'Humidity': 'mean',
            'Pressure': 'mean',
            'Wind Speed': 'mean',
            'Rain': 'sum'
        }
        resampled_df = wide_df.resample('1h').agg(agg_rules)
        
        # 3. Missing Value Imputation
        # Rain: Fill NaNs with 0 (assuming no report = no rain)
        resampled_df['Rain'] = resampled_df['Rain'].fillna(0.0)
        
        # Environmental Variables: Forward Fill up to 6 hours, then Backward Fill up to 6 hours
        weather_cols = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed']
        resampled_df[weather_cols] = resampled_df[weather_cols].ffill(limit=6)
        resampled_df[weather_cols] = resampled_df[weather_cols].bfill(limit=6)
        
        # If a sensor completely lacks a core feature (like 100% NaN for Wind Speed after filling), 
        # we mark it. For FSL, we can either drop the sensor or fill with regional means.
        # For simplicity, if Temperature or Humidity is entirely missing, we skip this sensor.
        if resampled_df['Temperature'].isna().all() or resampled_df['Humidity'].isna().all():
            print(f"Skipping {sensor}: Missing core environmental variables.")
            continue
            
        # For missing Wind Speed (very common), fill with 0 as a baseline assumption
        resampled_df['Wind Speed'] = resampled_df['Wind Speed'].fillna(0.0)
        
        # Add Sensor Name back as a column for context
        resampled_df = resampled_df.reset_index()
        resampled_df['Sensor_Name'] = sensor
        
        # Drop any remaining NaNs (e.g., long outages > 6 hours)
        resampled_df = resampled_df.dropna()
        
        if len(resampled_df) < 100:
            print(f"Skipping {sensor}: Too few valid hours ({len(resampled_df)}).")
            continue
            
        # 4. Output Delivery
        out_file = out_dir / f"{sensor}.parquet"
        resampled_df.to_parquet(out_file, index=False)
        processed_count += 1
        
    print(f"\nSuccessfully processed and saved {processed_count} sensors to {out_dir}/")

if __name__ == "__main__":
    raw_csv = "dataset/newcastle_rainfall_data.csv"
    output_dir = "dataset/processed"
    preprocess_sensor_data(raw_csv, output_dir)
