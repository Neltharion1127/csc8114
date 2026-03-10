import pandas as pd
import glob
import os

# 掃描資料夾
files = glob.glob("dataset/processed/*.parquet")
print(f"正在檢查 {len(files)} 個感測器的降雨數據...\n")

for f in files:
    df = pd.read_parquet(f)
    # 預估未來 3 小時降雨總量
    df['future_3h_rain'] = df['Rain'].shift(-3).rolling(window=3).sum()
    
    total_rows = len(df)
    rainy_rows = len(df[df['future_3h_rain'] > 0])
    
    status = "✅ 有雨" if rainy_rows > 0 else "❌ 全乾"
    print(f"感測器: {os.path.basename(f):<30} | 總筆數: {total_rows:<5} | 有雨片段: {rainy_rows:<5} | 狀態: {status}")

print("\n如果『有雨片段』為 0，模型就永遠無法在該感測器訓練 RAIN_SAMPLE。")