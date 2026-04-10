# 聯邦學習實驗檔案與儲存機制指南 (Experiment Storage Guide)

本文件詳述了 FSL 專案中 Server 與 Client 端的檔案儲存邏輯、命名規則及觸發時機，用於追蹤消融實驗 (Ablation Study) 的各項數據。

---

## 📋 核心機制彙整表

| 類別 | 檔案名稱範例 | 存檔時機 (Trigger) | 內容與用途 |
| :--- | :--- | :--- | :--- |
| **matrix_configs** | `{ts}_{scenario}_{seed}.yaml` | **實驗啟動前** (Mac 生成) | 原始Config，包含該場景所有超參數。 |
| **Client bestweights (weights)** | `best_client_{id}_round_{r}_{ts}.pth` | **每輪 server和client 同步完成後,會跑驗證.   若表現變好則存檔** | 該 Client 歷史表現最優的模型 (Best Model)。 |
| **Client periodic (weights)** | `periodic/client_{id}_round_{r}.pth` | **每輪 (Round) 結束後** (固定) | 目前config設定, 每一輪權重的定期備份，用於回溯或分析。 |
| **Client log (results)** | `training_log_client{id}_progress.csv` | **每一輪結束時** (Every Epoch) | **暫存版**進度紀錄。內容隨訓練累加，新實驗啟動時會覆寫。 seed52會覆蓋seed42的progress.csv |
| **Client final log (results)** | `training_log_client{id}_{ts}.csv` | **所有 Round 跑完時** (Final) | **Final版**完整紀錄。每一場 Seed 實驗跑完後產出的獨立檔案。 |
| **Client 摘要metadata (results)** | `..._meta.json` | **所有 Round 跑完時** | 實驗結果摘要 (Avg Metrics) 與Config setting。包含 Best Loss 等。 |
| **Server weights (weights)** | `server_head_round_{r}_{ts}.pth` | **每一輪聚合 (Aggregate) 後存檔** | Server 端更新完全局頭部模型後的權重存檔。 |
| **Server periodic (weights)** | `periodic/server_round_{r}.pth` | **每輪 (Round) 結束後** | Server 端全局權重的定期備份。 |
| **Server log (results)** | `server_log_{session_id}.csv` | **每一次前向傳播 (Forward)** | 每當有 Client 傳送資料過來，Server 算完 Loss 就會紀錄一筆。包含每個 Batch 的 Loss、處理延遲與壓縮比。 |

---

## ⚙️ 詳細運作邏輯

### 1. Client 端「最強權重」判定流程
Client 端的 `bestweights` 寫入遵循嚴格的判定邏輯（位在 `src/client/checkpointing.py`）：
1. **訓練 (Train)**：完成一輪局部訓練。
2. **同步 (Sync)**：與 Server 交換權重並聚合。
3. **驗證 (Validation)**：利用聚合後的模型在驗證集上跑一次推理。
4. **判定 (Evaluation)**：
   - 如果 **F1 Score** 創新高 $\rightarrow$ **存檔**。
   - 如果 F1 持平但 **MSE Loss** 創新低 $\rightarrow$ **存檔**。
   - 若皆無進步 $\rightarrow$ 增加 Early Stopping 計數器。

### 2. Results 日誌的覆寫與並存
*   **Progress 檔案**：檔名固定為 `progress.csv`。其目的是為了斷點續傳和實時監控，因此在新一場 Seed 實驗開始時會被初始化覆寫。
*   **日期 Final 檔案**：檔名包含 `{timestamp}`。這是實驗的正式成果，保證了 Seed 42 與 Seed 52 的數據可以同時存在於同一個場景資料夾內而不互相干擾。

### 3. Matrix Configs 的角色
`matrix_configs/` 資料夾下的 YAML 檔案是實驗的「身分證」。
* 每當執行 `make matrix` 時，系統會根據場景與種子的組合，將 `config.yaml` 拆解並分發。
* 這些檔案可用於事後比對：當某個場景結果異常時，開啟對應的 YAML 即可確認當時的硬體與軟體參數。

---

## 📂 資料夾路徑參考
* **設定檔**：`matrix_configs/*.yaml`
* **模型檔**：`bestweights/<session_id>/<scenario_id>/`
* **數據檔**：`results/<session_id>/<scenario_id>/`

---
*Last Updated: 2026-04-10*
