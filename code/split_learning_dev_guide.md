# Split Learning Guide (FSL Development Workflow)

這份指南包含了目前 FSL (Federated Split Learning) 專案從資料抓取到訓練評估的標準工作流。

## 1. 抓取與處理氣象資料 (Data Download)
我們目前已經棄用舊版的 Urban Observatory，改為使用 **Open-Meteo API** 來抓取精準且無缺失的歷史天氣與降雨資料。

```bash
uv run python src/data/data_download_openmeteo.py
```

**目的**：根據 `config.yaml` 裡設定的 `start_date` 和 `end_date`，自動抓取 Newcastle 範圍內 12 個感測器的資料，並直接處裡成訓練可用的序列檔案。
**輸出**：
`dataset/processed/*.parquet` (12份感測器乾淨資料，訓練即插即用)

> **💡 配置小技巧**：
> - 所有的資料時間範圍都在 `config.yaml` 中的 `data_download` 區塊控制。
> - `config.yaml` 裡的 `test_days: 14` 決定了每個 parquet 檔案裡最後 14 天的資料會被強制劃分為**測試集**，嚴格不參與訓練。

---

## 2. 啟動 FSL 網路與訓練 (Training & Evaluation)
不再需要手動起多個 terminal，現在完全由 Docker  orchestration 及 Makefile 接管。

```bash
make test-network
```

**這個指令會自動連貫執行：**
1. 建立最新的 Server 和 Client Docker Image。
2. 啟動 `fsl-server`，並透過 healthcheck 確保其 gRPC port 準備就緒。
3. 自動根據 `config.yaml` 裡的 `num_clients` (例如 3) 啟動對應數量的 `fsl-client`。
4. 在終端機上 tail (追蹤) 全部機器交錯的即時日誌。

### 訓練與評估生命週期：
每個 Epoch 的完整生命週期如下：
1. **[TRAIN] 邊緣端前向傳播**：Client 抓取訓練集資料，算出一小段**激活值 (Smashed Activation)**，依據 `compression: mode:` 設定壓縮後，發給 Server。
2. **[SERVER] 雲端反向傳播**：Server 收包、算 Loss、算梯度，再把梯度壓縮退回。
3. **[FED AVG] 全局聚合**：Client 跑完自己負責的資料後，發送權重到 Server 等待聚合。Server 收齊後合成最新全局模型下發。
4. **[EVALUATION] 獨立測試**：Client 拿到剛出爐的全局模型，用測試集（最後 14 天）跑前向傳播。此時附帶 `is_training=False` 標籤，Server **只算 Loss 不更新權重**，產出無偏見的測試成績。

---

## 3. 實驗與產出文件 (Logs & Results)

所有的實驗日誌都會被存放於本地的 `results/` 資料夾中（透過 Docker volume 映射出來）。

- **Server Log**：`results/server_log_*.csv`
  記錄每一筆傳輸的 Prediction、Target、Latency、資料壓縮解壓耗時等。
- **Client Log**：`results/training_log_client*.csv`
  記錄該 Client 每次前向傳播的 Loss 下降軌跡，附帶 `[TRAIN]` 或 `[TEST]` 狀態。

> **⚠️ 如何清理環境：**
> 如果程式卡死或想重新跑實驗，務必先清理舊容器：
> ```bash
> make clean
> ```

## 4. 視覺化分析 (Visualization)

### 4.1 Server Metrics Dashboard

```bash
# 載入最新 session 的 server log
uv run python src/data/plot_server_metrics.py

# 或指定 session
uv run python src/data/plot_server_metrics.py --session 20260312150555
```

**輸出**：`results/<session>/server_metrics_<session>.png`

### 4.2 Training Curve (Per-Client & Combined)

```bash
# 載入最新 session 的 client log
uv run python src/data/plot_training_curve.py

# 或指定 session
uv run python src/data/plot_training_curve.py --session 20260312150555
```

**輸出**：`results/<session>/training_curve_<session>.png`

---

## 5. 獨立評估 (Independent Evaluation)
當訓練完成後，我們可以使用獨立的評估腳本來檢驗模型在完全沒見過的 14 天測試集上的最終表現。

```bash
uv run python src/data/run_evaluation.py
```

**這個指令會執行：**
1. **模型自動選取**：自動從 `bestweights/` 中抓取最新的 Server 與對應的 Client 權重。
2. **數據標準化**：自動計算各感測器的 Mean/Std（與訓練時一致），確保預測基準正確。
3. **指標報告**：針對 12 個感測器分別輸出 MSE、MAE 以及降雨預測準確率（Accuracy）。

**輸出範例**：
你將會看到一個整合性表格，顯示各 Client 的表現以及全局平均值。這對於最終學術報告提供數據支持非常有用。

