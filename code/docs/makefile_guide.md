# Makefile 指令與啟動指南 (FSL Makefile Guide)

本文件整理了專案中常用的 Makefile 指令，以及啟動與分析實驗的標準流程。

---

## 1. Makefile 指令大全

| 指令 | 用途說明 | 備註 |
| :--- | :--- | :--- |
| `make compile-proto` | **協議編譯**：將 `.proto` 檔編譯成 Python 程式碼 | 修改網路通訊定義後必做 |
| `make download-data` | **資料下載**：從 Open-Meteo 下載歷史氣象資料 | 產出 Parquet 檔案 |
| `make run-native` | **本機一鍵啟動**：在本機自動啟動 Server 與所有 Clients | 開發調參最推薦，日誌更直觀 |
| **`make run-network`** | **Docker 一鍵啟動**：在 Docker 容器環境啟動全集群 | 模擬真實邊緣端網路環境 |
| `make plot-latest` | **快顯分析**：為最近一次實驗產出分析圖表 | 包含 Loss、Metrics 與混淆矩陣 |
| `make clean-native` | **清理本機進程**：強制殺死殘留的本機 Python 實驗進程 | 釋放 Port 50051 (127.0.0.1) |
| **`make clean`** | **清理 Docker**：移除所有 Docker 容器與虛擬網路 | 徹底重置模擬環境 |

---

## 2. 實驗啟動標準流程 (Standard Workflow)

這是開始一次新實驗的標準步驟：

1. **初始化**：`make compile-proto` (確保網路協議最新)
2. **資料準備**：`make download-data` (若已有資料可跳過)
3. **選擇環境**：
    - **本機開發調試 (推薦)**：執行 `make run-native`
    - **Docker 模擬測試**：執行 `make run-network`
4. **清理環境**：若實驗結束或出錯，務必執行以下指令釋放資源：
    - 本機清理：`make clean-native`
    - Docker 清理：`make clean`

---

## 3. Docker 環境操作 (模擬真實邊緣網路)

當您需要模擬多台機器在虛擬子網中通訊時，請使用：

*   **啟動環境**：
    ```bash
    make run-network
    ```
    這會讀取 `docker-compose.yml`，建立獨立的 `fsl-server` 容器與多個 `fsl-client` 容器。
*   **徹底清理**：
    ```bash
    make clean
    ```
    這會執行 `docker compose down -v`，移除所有容器、網路緩存以及資料卷。

---

## 4. 進階操作：本機手動分開執行 (適合 Debug)

如果您想在大終端機中分開觀察本機 Server 和個別 Client 的日誌：

### 步驟 A：啟動 Server (Terminal 1)
```bash
make run-native-server
```

### 步驟 B：啟動 所有 Clients (Terminal 2)
```bash
make run-native-client
```

### 步驟 C：啟動 單一特定的 Client (Terminal 3)
為了確保能抓到所有套件，建議使用 `.venv` 中的 python：
```bash
# 確保在 code 資料夾下執行
CLIENT_ID=1 FSL_SERVER_HOST=localhost .venv/bin/python -m src.nodes.client_node
```

---

## 5. 數據分析與繪圖 (Data Analysis)

實驗結束後，除了自動生成的圖表，您也可以手動執行以下指令進行深度分析：

### 方法一：使用 Makefile 快速繪圖
*   **針對最新一次實驗**：`make plot-latest`
*   **針對特定歷史 Session**：`make plot-session SESSION=<ID>`

### 方法二：手動執行 Python 指令 (更彈性)
*   **訓練曲線**：`python -m src.data.plot_training_curve --session <ID>`
*   **混淆矩陣**：`python -m src.data.plot_confusion_matrix --session <ID> --phase both`
*   **伺服器效能**：`python -m src.data.plot_server_metrics --log results/<ID>/server_log_<ID>.csv`
*   **最終跑分報告 (Evaluation)**：`.venv/bin/python -m src.data.run_evaluation` (自動加載最佳權重進行驗收)

---
*Last Updated: 2026-03-13*
