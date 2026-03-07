# uo-pyfetch is a lightweight Python library for fetching sensor data from an Urban Observatory Sensor API.  
# Instsll first
# pip install https://api.v2.urbanobservatory.ac.uk/lib/uo-pyfetch.tar.gz



from datetime import datetime, timedelta
import uo_pyfetch
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
last_days = 1
start_date = end_date - timedelta(days=2)

# --- Data Acquisition ---

print(f"Fetching meteorological data from {start_date.date()} to {end_date.date()}...")

# Retrieve sensor readings using the Urban Observatory pyfetch library
# last_n_days
# last_n_hours
raw_data = uo_pyfetch.get_sensor_data(
    variables=target_variables,
    bbox=newcastle_bbox,
    # start=start_date,
    end=end_date,
    last_n_days=last_days,
    limit=-1  # return all
)

# --- Data Processing & Storage ---
print(len(raw_data))
# Try this to see what the data actually looks like
if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
    print(f"Successfully retrieved {len(raw_data)} readings.")

    # Save to a CSV file
    # 'index=False' prevents Pandas from writing a row index column into the file
    filename = "newcastle_rainfall_data.csv"
    raw_data.to_csv(filename, index=False)
    print(f"Data has been successfully saved to: {filename}")
    print("\n--- Data Preview ---")
    print(raw_data.head())
else:
    print("No data found or the DataFrame is empty.")
