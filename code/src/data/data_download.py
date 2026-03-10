from datetime import datetime, timedelta
from uo_pyfetch import (
    get_sensors,
    get_sensor_data,
    get_sensor_data_by_name,
    get_variables,
    get_themes
)
import pandas as pd
from pathlib import Path
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Use shared config
from src.shared.common import cfg, project_root

# Read configuration values (with sensible defaults)
dd_cfg = cfg.get("data_download", {})

# Use data_download block for all download settings
newcastle_bbox = dd_cfg.get("bbox", [-1.71, 54.95, -1.54, 55.03])
target_variables = dd_cfg.get("feature_cols", ["Rain", "Temperature", "Humidity", "Pressure", "Wind Speed"])

# Date window (allow override via data_download.start_date / end_date)
end_date = datetime.fromisoformat(dd_cfg["end_date"]) if dd_cfg.get("end_date") else datetime.now()
if dd_cfg.get("start_date"):
    start_date = datetime.fromisoformat(dd_cfg["start_date"])
else:
    last_days = dd_cfg.get("last_days", 7)
    start_date = end_date - timedelta(days=last_days)

# Chunking and output
chunk_days = dd_cfg.get("chunk_days", 7)
out_dir = Path(project_root) / dd_cfg.get("raw_out_dir", "dataset")
out_dir.mkdir(parents=True, exist_ok=True)
filename = out_dir / dd_cfg.get("raw_filename", "newcastle_rainfall_data.csv")

# Remove existing file if starting fresh
if filename.exists():
    os.remove(filename)

# Data-download specific controls
retries = dd_cfg.get("retries", 3)
timeout_seconds = dd_cfg.get("timeout_seconds", 60)
sleep_seconds = dd_cfg.get("sleep_seconds", 2)

print(f"Fetching meteorological data from {start_date.date()} to {end_date.date()} in {chunk_days}-day chunks...")

current_end_date = end_date
total_rows = 0
chunk_count = 1

while current_end_date > start_date:
    current_start_date = max(start_date, current_end_date - timedelta(days=chunk_days))
    print(f"\n--- Fetching Chunk {chunk_count}: {current_start_date.date()} to {current_end_date.date()} ---")


    # Attempt with retries and per-call timeout
    attempt = 0
    raw_data = None
    last_exc = None
    while attempt < retries:
        attempt += 1
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_sensor_data,
                    variables=target_variables,
                    bbox=newcastle_bbox,
                    start=current_start_date,
                    end=current_end_date,
                    limit=-1, # return all
                )
                raw_data = future.result(timeout=timeout_seconds)
            break
        except FuturesTimeoutError:
            last_exc = f"timeout after {timeout_seconds}s"
            print(f"Chunk {chunk_count} attempt {attempt}: {last_exc}")
        except Exception as e:
            last_exc = str(e)
            print(f"Chunk {chunk_count} attempt {attempt} failed: {last_exc}")
        # exponential backoff between attempts
        if attempt < retries:
            time.sleep(min(30, 2 ** attempt))

    if raw_data is None:
        print(f"Failed to fetch chunk {chunk_count} ({current_start_date.date()} to {current_end_date.date()}) after {retries} attempts. Last error: {last_exc}")
    else:
        if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
            print(f"Successfully retrieved {len(raw_data)} readings in chunk {chunk_count}.")
            total_rows += len(raw_data)
            header = not filename.exists()
            raw_data.to_csv(filename, mode='a', index=False, header=header)
        else:
            print("No data found for this period.")

    current_end_date = current_start_date
    chunk_count += 1
    time.sleep(sleep_seconds)

print(f"\nFinished fetching. Total rows saved: {total_rows}")
print(f"Data has been successfully saved to: {filename}")
