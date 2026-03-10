import pandas as pd

# 讀取下載後的原始 CSV
df_raw = pd.read_csv("dataset/newcastle_rainfall_data.csv")

# 1. 看看有哪些感測器
print("目前抓到的感測器清單：")
print(df_raw['Sensor_Name'].unique())

# 2. 檢查 Rain 欄位的最大值
print(f"\n全檔案最大雨量: {df_raw['Value'][df_raw['Variable'] == 'Rain'].max()}")

# 3. 檢查 Rain > 0 的筆數
rain_data = df_raw[df_raw['Variable'] == 'Rain']
print(f"總雨量紀錄筆數: {len(rain_data)}")
print(f"有效降雨紀錄 (>0) 筆數: {len(rain_data[rain_data['Value'] > 0])}")