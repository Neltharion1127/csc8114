import pandas as pd
from pathlib import Path

data_dir = Path("dataset/processed")
files = list(data_dir.glob("NCL_*.parquet"))

print(f"{'Sensor Name':<20} | {'Total Hours':<12} | {'Rainy Hours':<12} | {'Rain %'}")
print("-" * 65)

for f in sorted(files):
    df = pd.read_parquet(f)
    total = len(df)
    rainy = (df["Rain"] > 0).sum()
    pct = (rainy / total) * 100
    print(f"{f.stem:<20} | {total:<12,} | {rainy:<12,} | {pct:.1f}%")