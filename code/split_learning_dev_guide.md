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

### 目前程式結構進度
為了改善 `client_node.py` / `server_node.py` 過長、難維護的問題，核心邏輯已經完成第一輪模組化重構：

- **Client side**
  - `src/client/data_pipeline.py`：資料載入與 train/test 採樣
  - `src/client/forward_step.py`：單步 split-learning 前向/反向流程
  - `src/client/sync.py`：FedAvg 同步
  - `src/client/reporting.py`：CSV / meta 輸出與 summary
  - `src/client/checkpointing.py`：best checkpoint、periodic checkpoint、early stopping
  - `src/client/scheduler_state.py`：train/test 分離的 scheduler state
- **Server side**
  - `src/server/scheduler.py`：壓縮模式分配邏輯
  - `src/server/forward_service.py`：Forward request 的實際運算與 structured logging
  - `src/server/fedavg.py`：FedAvg 聚合與 server checkpoint 管理
  - `src/server/reporting.py`：server log 落盤
  - `src/server/bootstrap.py`：gRPC server 啟動流程

目前 `src/nodes/client_node.py` 與 `src/nodes/server_node.py` 已經主要負責流程編排，不再承擔全部業務細節。

---

## 3. 實驗與產出文件 (Logs & Results)

所有的實驗日誌都會被存放於本地的 `results/<session>/` 資料夾中（透過 Docker volume 映射出來）。

- **Server Log**：`results/<session>/server_log_<session>.csv`
  記錄每一筆傳輸的 Prediction、Target、Latency、compression mode、解壓/計算耗時等。
- **Client Log**：`results/<session>/training_log_client*.csv`
  記錄該 Client 每次前向傳播的 Loss、Latency、PayloadBytes、NextCompression，附帶 `[TRAIN]` 或 `[TEST]` 狀態。
- **Client Meta**：`results/<session>/training_log_client*_meta.json`
  摘要保存 `best_test_loss`、`best_model_path`、平均 latency / payload，以及當下完整 `cfg`。

### Scheduler / Profiler 目前狀態
`config.yaml` 裡以下設定現在都已經接入程式，不再只是預留欄位：

```yaml
profiler:
  enabled: true

scheduler:
  enabled: true
  latency_threshold: 4.0
  int8_latency_threshold: 10.0
```

- `profiler.enabled`
  - 關閉後，client 不再回報真實 `latency_ms` / `payload_bytes`
  - client log 內的 `LatencyMs` 與 `PayloadBytes` 也會記為 `0`
- `scheduler.enabled`
  - 關閉後，server 不再動態切換 compression mode
  - client 會維持原本 `compression.mode`
- `scheduler.latency_threshold`
  - 控制切到 `float16` 的閾值
- `scheduler.int8_latency_threshold`
  - 控制切到 `int8` 的閾值

目前 `rho` 相關設定尚未接入 scheduler 行為，保留到下一階段再處理。

### Log 欄位補充
為了避免把「feature 關閉」誤判成「真實延遲接近 0」，目前 client / server 日誌都會額外記錄：

- `ProfilerEnabled`
- `SchedulerEnabled`

server log 也會同步落下：

- `profiler_enabled`
- `scheduler_enabled`

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

---

## 6. 目前實驗結論與注意事項

### 已確認正常的部分
- FSL train / test / FedAvg / logging 全鏈路可以正常跑完
- Scheduler 功能上已正常工作，server log 中可觀察到 `float32 / float16 / int8` 的切換
- Profiler 與 Scheduler 已有獨立 config 開關，且狀態會寫入 log
- Server checkpoint 清理的排序 bug 已修正，不再因字典序錯誤保留舊 checkpoint

### 目前尚未證明有效的部分
- **模型訓練品質仍不理想**
  最新一輪實驗中，模型 prediction 有明顯接近「全部輸出 0」的現象，因此：
  - `best_test_loss = 0.0` 不代表模型真的學好
  - 可能只是抽到 `target = 0` 的測試樣本時剛好命中
- 因此目前可以說：
  - **scheduler 在工作**
  - 但還不能說 **scheduler 已經幫助模型學得更好**

### 下一階段建議
1. 針對「prediction collapse to 0」做專門排查
2. 擴充 evaluation，不只看單次抽樣的 test loss
3. 再評估是否需要引入更進階的 scheduler / rho 控制策略

---

## 7. 常見排錯 (Troubleshooting)

### 7.1 為什麼 `best_test_loss = 0.0`，但我還是不能說模型訓練成功？
這通常代表：

- 某次測試抽樣剛好抽到 `target = 0`
- 模型 prediction 也剛好是 `0`
- 該次單點 loss 因此變成 `0`

這不等於模型已經學會降雨模式。要同時搭配以下資訊一起判斷：

- `training_log_client*.csv` 裡整體 `TEST` loss 的平均值
- prediction 是否大量接近 `0`
- rain / dry classification accuracy 是否明顯高於隨機

如果你看到：

- `best_test_loss = 0.0`
- 但 prediction 幾乎全是 `0`
- 而且 accuracy 接近 `50%`

那更可能是 **prediction collapse**，不是訓練成功。

### 7.2 為什麼 log 裡的 `LatencyMs` 或 `PayloadBytes` 全是 `0`？
先不要直接判定 profiler 壞掉，先看開關：

- client log 欄位：
  - `ProfilerEnabled`
  - `SchedulerEnabled`
- client meta 欄位：
  - `profiler_enabled`
  - `scheduler_enabled`
- server log 欄位：
  - `profiler_enabled`
  - `scheduler_enabled`

若 `profiler_enabled = false`：

- client 不會回報真實 `latency_ms`
- `payload_bytes` 也會寫成 `0`
- 這是預期行為，不是 bug

### 7.3 怎麼確認 scheduler 真的有在工作？
不要只看 console print，請直接檢查 `results/<session>/server_log_<session>.csv`：

- `compression_mode`
- `next_compression`
- `reported_latency_ms`

如果 scheduler 正常工作，通常會看到：

- `compression_mode` 不只一種值
- `next_compression` 會在 `float32 / float16 / int8` 之間切換
- 切換和 `reported_latency_ms` 有對應關係

如果你只看到：

- `compression_mode = float32`
- `next_compression = float32`

那常見原因有兩種：

1. `scheduler.enabled = false`
2. `reported_latency_ms` 從未超過 threshold，所以 scheduler 沒有切換的必要

### 7.4 為什麼 scheduler 看起來正常，但模型效果還是很差？
這兩件事要分開看：

- **scheduler 正常工作**
  代表 compression mode 會依延遲切換
- **模型訓練有效**
  代表 prediction 與 target 關係合理，loss 和 accuracy 有改善

目前專案中，已確認前者成立，但後者仍需進一步驗證。換句話說：

- `scheduler works`
- 不等於 `training works`

### 7.5 為什麼 bestweights 目錄裡的 checkpoint 看起來比實際 round 小？
這個問題曾經出現在 server checkpoint 清理邏輯中，原因是舊版按**字串排序**而不是按 **round 數值排序** 清理檔案。

目前這個 bug 已修正。若之後再次看到異常：

- 請先確認是不是在看 `server_head_round_*`
- 再確認 `periodic/server_round_*.pth` 是否與 server log 的 `round` 對得上
