from datetime import datetime, timedelta
from uo_pyfetch import (
    get_sensors,
    get_sensor_data,
    get_sensor_data_by_name,
    get_variables,
    get_themes
)
import pandas as pd

# --- Configuration & Parameters ---
# Define the spatial boundary for Newcastle (p1_x, p1_y, p2_x, p2_y)
newcastle_bbox = [-1.71, 54.95, -1.54, 55.03] 

# Key meteorological predictors for rainfall forecasting
target_variables = [
    "Rain",           # Target: Accumulated precipitation
    "Temperature",    # Feature: Influence on air saturation and frontal movements
    "Humidity",  # Feature: Direct indicator of moisture levels
    "Pressure",           # Feature: Low-pressure systems often precede rainfall
    "Wind Speed"          # Feature: Affects the movement of weather systems
]


# Set the historical data window (e.g., the last 30 days)
end_date = datetime.now()
last_days = 365
start_date = end_date - timedelta(days=365)

# --- Data Acquisition in Chunks ---

from pathlib import Path
import os
import time

out_dir = Path("dataset")
out_dir.mkdir(parents=True, exist_ok=True)
filename = out_dir / "newcastle_rainfall_data.csv"

# Remove existing file if starting fresh
if filename.exists():
    os.remove(filename)

print(f"Fetching meteorological data from {start_date.date()} to {end_date.date()} in 30-day chunks...")

# Generate a list of dates to iterate through (chunking by ~30 days)
current_end_date = end_date
chunk_days = 30

total_rows = 0
chunk_count = 1

while current_end_date > start_date:
    current_start_date = max(start_date, current_end_date - timedelta(days=chunk_days))
    
    # We use 'start' and 'end' parameters instead of last_n_days to fetch specific windows
    actual_days_in_chunk = (current_end_date - current_start_date).days
    print(f"\n--- Fetching Chunk {chunk_count}: {current_start_date.date()} to {current_end_date.date()} ---")
    
    try:
        raw_data = get_sensor_data(
            variables=target_variables,
            bbox=newcastle_bbox,
            start=current_start_date,
            end=current_end_date,
            limit=-1  # return all
        )
        
        if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
            print(f"Successfully retrieved {len(raw_data)} readings in chunk {chunk_count}.")
            total_rows += len(raw_data)
            
            # Append to CSV. Write header only on the first chunk.
            header = not filename.exists()
            raw_data.to_csv(filename, mode='a', index=False, header=header)
            
        else:
            print("No data found for this period.")
            
    except Exception as e:
        print(f"Failed to fetch chunk {chunk_count}: {e}")
        
    current_end_date = current_start_date
    chunk_count += 1
    
    # Sleep to be polite to the API server
    time.sleep(2)

print(f"\nFinished fetching. Total rows saved: {total_rows}")
print(f"Data has been successfully saved to: {filename}")
