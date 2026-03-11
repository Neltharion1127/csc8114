"""
data_download_openmeteo.py

Downloads hourly weather data for 12 locations around Newcastle upon Tyne
from the Open-Meteo Historical Weather API (https://open-meteo.com/).
No API key required.

Output: one parquet file per location in dataset/processed/,
        matching the format expected by client_node.py:
        Columns: Timestamp, Temperature, Humidity, Pressure, Wind Speed, Rain, Sensor_Name
"""

import requests
import pandas as pd
from pathlib import Path
import os
import sys

# Use shared config
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.shared.common import cfg, project_root
from datetime import datetime

# --- Configuration ---
dd_cfg = cfg.get("data_download", {})

# Read dates from config, default to 2025-01-01 to today if missing
try:
    start_dt = datetime.fromisoformat(dd_cfg.get("start_date", "2025-01-01T00:00:00"))
    START_DATE = start_dt.strftime("%Y-%m-%d")
except (ValueError, TypeError):
    START_DATE = "2025-01-01"

if dd_cfg.get("end_date"):
    try:
        end_dt = datetime.fromisoformat(dd_cfg["end_date"])
        END_DATE = end_dt.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        END_DATE = datetime.now().strftime("%Y-%m-%d")
else:
    END_DATE = datetime.now().strftime("%Y-%m-%d")

OUT_DIR = Path(project_root) / dd_cfg.get("raw_out_dir", "dataset") / "processed"

# Open-Meteo API endpoint for historical weather re-analysis (ERA5)
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# 12 representative locations around Newcastle to simulate distributed sensors
LOCATIONS = [
    {"name": "NCL_CITY_CENTRE",  "lat": 54.978,  "lon": -1.617},
    {"name": "NCL_JESMOND",      "lat": 54.988,  "lon": -1.602},
    {"name": "NCL_GATESHEAD",    "lat": 54.962,  "lon": -1.601},
    {"name": "NCL_GOSFORTH",     "lat": 55.001,  "lon": -1.616},
    {"name": "NCL_WALLSEND",     "lat": 55.000,  "lon": -1.534},
    {"name": "NCL_BYKER",        "lat": 54.972,  "lon": -1.574},
    {"name": "NCL_HEATON",       "lat": 54.980,  "lon": -1.570},
    {"name": "NCL_FENHAM",       "lat": 54.984,  "lon": -1.645},
    {"name": "NCL_WALKER",       "lat": 54.974,  "lon": -1.551},
    {"name": "NCL_BLAYDON",      "lat": 54.966,  "lon": -1.712},
    {"name": "NCL_SCOTSWOOD",    "lat": 54.970,  "lon": -1.654},
    {"name": "NCL_BENWELL",      "lat": 54.975,  "lon": -1.638},
]

# Open-Meteo variable mapping → our column names
VARIABLE_MAP = {
    "temperature_2m":         "Temperature",
    "relative_humidity_2m":   "Humidity",
    "surface_pressure":       "Pressure",
    "wind_speed_10m":         "Wind Speed",
    "rain":                   "Rain",
}


def fetch_location(loc: dict) -> pd.DataFrame | None:
    """Fetch hourly weather for one location from Open-Meteo."""
    params = {
        "latitude":   loc["lat"],
        "longitude":  loc["lon"],
        "start_date": START_DATE,
        "end_date":   END_DATE,
        "hourly":     list(VARIABLE_MAP.keys()),
        "timezone":   "Europe/London",
    }

    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "Timestamp":   pd.to_datetime(hourly["time"]),
        "Temperature": hourly["temperature_2m"],
        "Humidity":    hourly["relative_humidity_2m"],
        "Pressure":    hourly["surface_pressure"],
        "Wind Speed":  hourly["wind_speed_10m"],
        "Rain":        hourly["rain"],
        "Sensor_Name": loc["name"],
    })

    # Drop rows where any core feature is missing
    df = df.dropna(subset=["Temperature", "Humidity", "Pressure", "Rain"])

    # Rain: fill remaining NaN with 0 (no rain = 0mm)
    df["Rain"] = df["Rain"].fillna(0.0)
    df["Wind Speed"] = df["Wind Speed"].fillna(0.0)

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Open-Meteo data for {len(LOCATIONS)} locations")
    print(f"Period: {START_DATE} → {END_DATE}")
    print(f"Output: {OUT_DIR}\n")

    for loc in LOCATIONS:
        print(f"  [{loc['name']}] Fetching...", end=" ", flush=True)
        try:
            df = fetch_location(loc)
            out_path = OUT_DIR / f"{loc['name']}.parquet"
            df.to_parquet(out_path, index=False)
            rain_rows = (df["Rain"] > 0).sum()
            print(f"✓  {len(df):,} rows | Rain: {rain_rows:,} samples ({100*rain_rows/len(df):.1f}%)")
        except Exception as e:
            print(f"✗  FAILED: {e}")

    # Quick summary
    files = list(OUT_DIR.glob("NCL_*.parquet"))
    print(f"\n✅ Done. {len(files)}/{len(LOCATIONS)} locations saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
