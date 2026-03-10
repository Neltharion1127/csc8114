import pandas as pd
import os
import glob
from src.shared.common import project_root

def standardize_all_sensors():
    # 1. 路徑設定
    raw_csv = os.path.join(project_root, "dataset/newcastle_rainfall_data.csv")
    processed_dir = os.path.join(project_root, "dataset/processed")
    os.makedirs(processed_dir, exist_ok=True)

    print(f"正在從原始 CSV 讀取數據：{raw_csv}")
    df = pd.read_csv(raw_csv)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # 2. 取得所有感測器清單
    sensors = df['Sensor_Name'].unique()
    print(f"發現 {len(sensors)} 個感測器，開始進行 h 重採樣標準化...\n")

    for sensor in sensors:
        try:
            # 過濾特定感測器數據
            sensor_df = df[df['Sensor_Name'] == sensor].copy()
            
            # 轉換為樞紐表 (Pivot) 將 Variable 轉為列
            pivoted = sensor_df.pivot_table(index='Timestamp', columns='Variable', values='Value')
            
            # 3. 執行小時重採樣 (1h Resampling)
            # 對於 Rain (降雨) 使用 sum() 累加；對於溫度等其他特徵使用 mean() 平均
            resampled = pd.DataFrame()
            
            if 'Rain' in pivoted.columns:
                resampled['Rain'] = pivoted['Rain'].resample('1h').sum()
            
            # 處理其他氣象特徵 (使用平均值)
            other_cols = [c for c in pivoted.columns if c != 'Rain']
            for col in other_cols:
                resampled[col] = pivoted[col].resample('1h').mean()

            # 4. 填補缺失值與特徵對齊
            # 確保 24 小時訓練所需的基本特徵都存在，缺少的補 0.0
            required_features = ["Temperature", "Humidity", "Pressure", "Wind Speed", "Rain"]
            for feat in required_features:
                if feat not in resampled.columns:
                    resampled[feat] = 0.0
            
            # 處理 NaN (先線性插值，最後補 0)
            resampled = resampled.interpolate(method='linear').fillna(0.0)

            # 5. 儲存為 Parquet
            out_name = f"{sensor}.parquet"
            resampled.to_parquet(os.path.join(processed_dir, out_name))
            
            # 診斷輸出
            rain_val = resampled['Rain'].max()
            print(f"{sensor[:25]:<25} | 筆數: {len(resampled):<5} | 最大雨量: {rain_val:.2f}")

        except Exception as e:
            print(f"{sensor} Error: {str(e)}")

    print(f"\n✨ 所有感測器已標準化為 1 小時頻率，存於 {processed_dir}")

if __name__ == "__main__":
    standardize_all_sensors()