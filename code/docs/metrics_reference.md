# 聯邦學習實驗指標記錄總覽

> 更新時間：2026-04-29（含 `total_payload_mb` 最新修改）

---

## 1. 每步訓練日誌 `training_log_client[ID].csv`

每一次 Forward Pass 寫入一列，每台 Pi 獨立產生。

| 欄位 | 型態 | 說明 | 論文用途 |
|---|---|---|---|
| `Epoch` | int | 第幾輪 | — |
| `Status` | str | `TRAIN` / `VAL` | — |
| `Sensor` | str | 氣象站 ID | — |
| `Target` | float | 真實降雨量 (mm) | — |
| `Prediction` | float | 模型預測值 | — |
| `RainFlag` | int | 是否下雨 (0/1) | — |
| `Loss` | float | 總損失值 | — |
| `RainProbability` | float | 降雨機率 (0~1) | — |
| `ClassificationLoss` | float | 分類損失 | — |
| `RegressionLoss` | float | 回歸損失 | — |
| `LatencyMs` | float | 每步 gRPC 往返延遲 (ms) | ✅ 通訊延遲分析 |
| `PayloadBytes` | int | 壓縮後 Activation 大小 (bytes) | ✅ 核心傳輸量指標 |
| `CompressionMode` | str | **此步實際使用的壓縮模式** | ✅ 壓縮階梯圖 |
| `SparsityRatio` | float | **Top-K 傳送比例** (1.0 = 全部) | ✅ 稀疏度分析 |
| `NextCompression` | str | Scheduler 決定下一步的模式 | ✅ 自適應行為追蹤 |
| `NextRho` | int | Scheduler 決定下一步的 ρ | ✅ 同步頻率追蹤 |
| `CPU_Percent` | float | 當輪 CPU 使用率 (%) | ✅ 硬體負載 |
| `Mem_Percent` | float | 當輪記憶體使用率 (%) | ✅ 硬體負載 |
| `Mem_RSS_MB` | float | 當輪實體記憶體 RSS (MB) | ✅ 記憶體追蹤 |
| `Mem_Peak_MB` | float | 到此為止的記憶體峰值 (MB) | ✅ 峰值分析 |
| `Net_Sent_MB` | float | 累計網卡發送量 (MB) | ✅ 實體網路流量 |
| `Net_Recv_MB` | float | 累計網卡接收量 (MB) | ✅ 實體網路流量 |
| `Epoch_Time_s` | float | 當輪訓練耗時 (s) | ✅ 效率分析 |
| `Model_Size_Bytes` | int | Client 模型大小 (bytes) | 參考值 |

---

## 2. 訓練摘要 `training_log_client[ID]_meta.json`

整個訓練結束後，每台 Pi 獨立產生一個 JSON 彙總檔。

| 欄位 | 說明 | 論文用途 |
|---|---|---|
| `avg_latency_ms` | 平均每步 gRPC 延遲 (ms) | ✅ 通訊延遲 |
| `avg_payload_bytes` | 平均每步傳輸量 (bytes) | ✅ 核心傳輸量 |
| `num_records` | 總訓練步數 | 計算用 |
| `avg_cpu_percent` | 平均 CPU 使用率 (%) | ✅ 硬體負載 |
| `avg_mem_percent` | 平均記憶體使用率 (%) | ✅ 硬體負載 |
| `total_runtime_s` | 總訓練時間 (s) | ✅ 效率分析 |
| `model_size_bytes` | Client 模型大小 (bytes) | 參考值 |
| `net_sent_mb` | 訓練期間網卡發送總量 (MB) | ✅ 實體網路流量 |
| `net_recv_mb` | 訓練期間網卡接收總量 (MB) | ✅ 實體網路流量 |
| `mem_peak_mb` | 記憶體使用峰值 (MB) | ✅ 記憶體峰值 |
| `sync_bytes_sent_mb` | FedAvg 模型上傳總量 (MB) | ✅ 模型同步傳輸量 |
| `sync_bytes_recv_mb` | FedAvg 全局模型下載總量 (MB) | ✅ 模型同步傳輸量 |

---

## 3. 評估報告 `[scenario]_eval_report.csv`

執行 `make eval-latest` / `make eval-session` 後產生。  
每個 Client 一列，最後一列為 `SUMMARY`（全集群彙總）。

### 3a. 效能指標（每個 Client）

| 欄位 | 說明 |
|---|---|
| `client_id` | Client 編號（SUMMARY 列顯示 `SUMMARY`）|
| `best_round` | 最佳 FedAvg Round |
| `train_loss` | 最佳訓練損失 |
| `samples` | 測試樣本數 |
| `mse` | 均方誤差 |
| `mae` | 平均絕對誤差 (mm) |
| `accuracy` | 分類準確率 |
| `f1` | F1-Score |
| `recall` | 召回率 |
| `precision` | 精確率 |
| `auprc` | PR-AUC（主要指標）|
| `roc_auc` | ROC-AUC |
| `brier` | Brier Score |
| `tp / fn / fp / tn` | 混淆矩陣 |
| `prob_threshold` | 使用的機率門檻 |
| `monthly_details` | 各月份 MSE / Acc / F1 / Samples（JSON，最後一欄）|

### 3b. 硬體遙測指標

| 欄位 | 說明 | 個別 Client | SUMMARY |
|---|---|---|---|
| `cpu_percent` | 平均 CPU 使用率 (%) | ✅ 各 Pi | 平均 |
| `mem_percent` | 平均記憶體使用率 (%) | ✅ 各 Pi | 平均 |
| `runtime_s` | 訓練總時間 (s) | ✅ 各 Pi | 平均 |
| `model_size_mb` | Client 模型大小 (MB) | ✅ 各 Pi | — |
| `payload_bytes` | 平均每步傳輸量 (bytes) | ✅ 各 Pi | — |
| `latency_ms` | 平均 gRPC 延遲 (ms) | ✅ 各 Pi | — |
| `throughput_sps` | 樣本吞吐量 (samples/s) | ✅ 各 Pi | 總樣本/平均時間 |
| `data_throughput_kbps` | 數據量吞吐量 (KB/s) | ✅ 各 Pi | 總傳輸量/平均時間 |
| `total_payload_mb` | **全訓練 Payload 總量 (MB)** | ✅ 各 Pi | 全集群加總 |
| `net_sent_mb` | 實體網卡發送總量 (MB) | ✅ 各 Pi | 全集群加總 |
| `net_recv_mb` | 實體網卡接收總量 (MB) | ✅ 各 Pi | 全集群加總 |
| `mem_peak_mb` | 記憶體使用峰值 (MB) | ✅ 各 Pi | 平均峰值 |
| `sync_bytes_sent_mb` | FedAvg 上傳量 (MB) | ✅ 各 Pi | 全集群加總 |
| `sync_bytes_recv_mb` | FedAvg 下載量 (MB) | ✅ 各 Pi | 全集群加總 |

---

## 4. 與會議文件 Results Table 的對應關係

| Results Table 欄位 | CSV 對應欄位 | 說明 |
|---|---|---|
| F1 | `f1` (SUMMARY) | 加權平均 F1 |
| AUC | `roc_auc` (SUMMARY) | 加權平均 ROC-AUC |
| Acc | `accuracy` (SUMMARY) | 加權平均準確率 |
| MSE | `mse` (SUMMARY) | 加權平均 MSE |
| MAE | `mae` (SUMMARY) | 加權平均 MAE |
| Thrp | `throughput_sps` (SUMMARY) | 全集群樣本吞吐量 |
| Size | `model_size_mb` | 每台 Pi 相同（架構固定）|
| **Traff (payload)** | **`total_payload_mb` (SUMMARY)** | 全集群 Payload 傳輸量 |
| CPU/M | `cpu_percent` / `mem_percent` (SUMMARY) | 平均 CPU / 記憶體 |

---

## 5. 壓縮模式對照（每步訓練日誌分析）

| Scenario | `CompressionMode` 變化 | `SparsityRatio` | `PayloadBytes` 預期 |
|---|---|---|---|
| PI01 (float32) | 全程 `float32` | `1.0` | hidden × 4 bytes = ~256 B |
| PI02 (float16) | 全程 `float16` | `1.0` | hidden × 2 bytes = ~128 B |
| PI03 (rho=4) | 全程 `float32` | `1.0` | ~256 B，但同步次數 ÷4 |
| PI04 (adaptive) | `float32` → `float16` → `int8` 動態切換 | `1.0`（除非 topk）| 階梯式下降 |

> `CompressionMode` + `PayloadBytes` 時間序列 = 「階梯式壓縮」的直接證據  
> `total_payload_mb` (SUMMARY) = Results Table 的 **Traff (payload)** 填表數值
