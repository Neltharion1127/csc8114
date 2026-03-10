
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
- 確保 xxx_EA_TPRG 檔案已生成且包含雨量數據
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

## 4. 啟動 Server / Client（Docker）
```bash
make test-network
```

**目的**：使用 Docker Compose 啟動 server / client 模組，建立完整 SplitNN 網路。

---

## （Optional）手動啟動 Server / Client

### (1)啟動 Server（接收 smashed activations）
- 檢查設定：打開 config.yaml，確認 num_clients 的數量（例如 3）
- 啟動 server
```bash
uv run python -m src.nodes.server_node
```
- Server 生成檔案：`results/server_log_*.csv` 記錄所有 Client 傳來的 Loss、預測值 (Prediction) 與真實降雨量 (Target)。

### (2)啟動 Client 
- 開啟與 num_clients 數量相同的終端機視窗，分別輸入對應 ID
視窗 1, 視窗 2, 視窗 3, ...
```bash
uv run python -m src.nodes.client_node 1
uv run python -m src.nodes.client_node 2
uv run python -m src.nodes.client_node 3
```
- Client 生成檔案：`results/training_log_client{id}_*.csv` 記錄該 Client 負責的感測器訓練細節、延遲 (Latency) 與資料傳輸量 (Payload)

**Hint**
- 避免卡死：Server 會等待所有 Client 到齊才開始聚合權重。若只想開 1 個視窗測試，請將 num_clients 改為 1。
- 數據分配：每個 Client 會根據 ID 自動領取 1/N 的 Parquet 檔案，確保雨量計數據不遺漏。

**預期結果**：
- 在訓練過程中應看到 **`[RAIN_SAMPL]`** 字樣  
- 代表模型正在針對「有雨片段」進行強化學習

---
