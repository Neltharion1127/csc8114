
# Split Learning Guide（Data → Preprocess → Check → Train）

## 1. 下載原始雨量資料  
使用 Urban Observatory API 抓取 Newcastle 本地感測器資料。

```bash
uv run python src/data/data_download.py
```

**目的**：從 Newcastle Urban Observatory 抓取原始 CSV 資料。  
**輸出**：`dataset/newcastle_rainfall_data.csv`

**補充**: 下載的時間範圍可在 `config.yaml` 中的 `data_download` 區塊手動設定

---

## 2. 標準化與切割雨量資料（取代舊 preprocessing）
```bash
uv run python src/data/standardize_rainfalldata.py
```

**目的：**
- 依感測器類型拆分原始 CSV
- 將不同頻率（例如 15 分鐘）重採樣為 **1 小時**
- 輸出乾淨、可用的 parquet 格式資料

**輸出**：  
`dataset/processed/*.parquet`  
（包含重要的 **EA_TPRG** 感測器檔案）

---

## 3. 資料品質檢查（非常重要）
```bash
uv run python src/data/senseor_data_check.py
```

**目的：**
- 確認 **EA_TPRG** 感測器是否顯示「✅ 有雨」
- 確保雨量片段數量 > 0

> ⚠️ 若此步驟看不到雨，後續模型將 **無法學會降雨預測**

---

## 4. 啟動 Split Learning 環境（Docker）
```bash
make test-network
```

**目的**：使用 Docker Compose 啟動 server / client 模組，建立完整 SplitNN 網路。

---

## （Optional）4-2 手動啟動 Server / Client

## 啟動 Server（接收 smashed activations）
```bash
uv run python src/server/server_node.py
```

## 啟動 Client（進行 split learning 訓練）
```bash
uv run python src/client/client_node.py
```

**預期結果**：
- 在訓練過程中應看到 **`[RAIN_EVENT]`** 字樣  
- 代表模型正在針對「有雨片段」進行強化學習

---
