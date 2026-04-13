# FSL Experiment Runbook / 实验运行手册

---

## Prerequisites / 前置条件

| Requirement / 条件 | Check / 检查方式 |
|---|---|
| Tailscale connected / Tailscale 已连接 | `tailscale status` |
| Ansible installed / Ansible 已安装 | `ansible --version` |
| VPS reachable / VPS 可达 | `ping 51.254.207.168` |
| Working directory / 工作目录 | All `make` commands run from `code/` |

---

## 1. Build & Push Image / 构建并推送镜像

Only needed when `src/`, `Dockerfile`, `pyproject.toml`, or `proto/` change.
仅在修改了 `src/`、`Dockerfile`、`pyproject.toml` 或 `proto/` 时需要。

```bash
git add .
git commit -m "..."
git push origin main
```

GitHub Actions builds a multi-arch image (amd64 + arm64) and pushes to Docker Hub automatically (~5–10 min).
GitHub Actions 自动构建双架构镜像并推送，等待约 5–10 分钟。

---

## 2. Deploy Image to Pis / 分发镜像到 Pi

Pi 通常无法直接访问 Docker Hub（TLS timeout），用以下方式之一分发：

**Option A — Build on Mac and push directly (recommended) / Mac 本地构建后直接推（推荐）：**
```bash
make dist-load-image-local
```

**Option B — Route through VPS / 通过 VPS 中转：**
```bash
make dist-load-image
```

---

## 3. Fresh Start (most common) / 全新启动实验（最常用）

一条命令完成：sync config → 重启服务端 → 等 gRPC 就绪 → 部署所有 Pi 客户端。

```bash
make dist-start
```

---

## 4. Nuclear Restart / 核弹重启

停止所有训练 + 清空所有结果 + 全新启动。

```bash
make dist-restart
```

---

## 5. Monitor Progress / 查看进度

**VPS server logs / 服务端日志：**
```bash
make dist-logs
```

**Single Pi client log / 单台 Pi 客户端日志：**
```bash
ssh pi@100.125.109.94 "docker logs -f fsl-client"
```

**All Pi container status / 所有 Pi 容器状态：**
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -m shell \
  -a "docker ps -a | grep fsl-client" --become
```

---

## 6. Config-only Update / 仅更新配置

代码不变，只改了 config.yaml：

```bash
make dist-sync-config     # 推 config 到 VPS + 所有 Pi
make dist-server-restart  # 重启服务端使配置生效
# 然后重新部署客户端
ansible-playbook ansible/deploy_client.yml -i ansible/inventory.taisale.ini
```

Or all-in-one / 或一条命令：
```bash
make dist-start
```

---

## 7. Clean Results / 清空结果

```bash
make dist-clean-server   # 清空 VPS 的 results/ 和 bestweights/
make dist-clean-results  # 清空所有 Pi 的 results/ 和 bestweights/
```

---

## 8. Collect Results / 收集实验结果

**Pull from VPS / 从 VPS 拉取：**
```bash
scp -r ubuntu@51.254.207.168:~/csc8114/code/results/ code/results/vps_results/
scp -r ubuntu@51.254.207.168:~/csc8114/code/bestweights/ code/bestweights/
```
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -m ansible.posix.synchronize \
  -a "mode=pull src=/home/pi/bestweights/ dest=code/bestweights/"
```

**Pull from all Pis / 从所有 Pi 拉取：**
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -m ansible.posix.synchronize \
  -a "mode=pull src=/home/pi/results/ dest=code/results/pi_results/"
```

---

## 9. Debug: Check Why a Pi Failed / 排查某台 Pi 异常

```bash
# 查看容器退出状态
ansible clients -i ansible/inventory.taisale.ini \
  -m shell -a "docker ps -a | grep fsl-client" --become

# 查看某台 Pi 的容器日志
ansible pi08 -i ansible/inventory.taisale.ini \
  -m shell -a "docker logs fsl-client 2>&1 | tail -30" --become

# 常见错误
# "exec format error" → 镜像架构错误（amd64 跑在 arm64 Pi 上），需重新 make dist-load-image-local
# "connection refused"  → 服务端未启动或网络不通
```

---

## Quick Reference / 快速参考

| Action / 操作 | Command / 命令 |
|---|---|
| Trigger image build / 触发镜像构建 | `git push origin main` |
| Start server / 启动服务端 | VPS: `docker compose -f docker-compose.server.yml up -d` |
| Deploy all clients / 部署所有客户端 | `ansible-playbook ansible/deploy_client.yml -i ansible/inventory.ini` |
| Watch server logs / 看服务端日志 | `docker logs -f fsl-server` |
| Stop all clients / 停止所有客户端 | `ansible clients -i ansible/inventory.ini -a "docker stop fsl-client" --become` |
| Deploy to subset of Pis / 只部署部分 Pi | append `--limit pi01,pi02` to ansible command |
| Run specific matrix scenarios / 跑特定场景 | `python -m src.data.run_experiment_matrix --only M06,M07` |
| Get all client's containers status/ 查看所有正在執行的容器|`ansible all -i ansible/inventory.ini -b -a "docker ps"`|
| Get all client's directories/ 查看所有client的目錄|`ansible all -i ansible/inventory.ini -b -a "ls -lh /home/pi"`|
| Clean all client's docker images/ 清理所有client的 docker鏡像|`ansible clients -i ansible/inventory.ini -m shell -a "docker system prune -a -f --volumes" --become`|
| Clean VPS docker images/ 清理VPS的 docker鏡像|`ssh ubuntu@51.254.207.168 "docker system prune -a -f --volumes"`|
||make dist-clean-results|
||make dist-clean-server|

---

# 從所有 Raspberry Pi 拉回 Client 端結果與模型
## 拉回日誌與結果
ansible clients -i ansible/inventory.ini \
  -m ansible.posix.synchronize \
  -a "mode=pull src=/home/pi/results/ dest=results/pi_results/"

## 拉回 Client 端權重 (排除periodic 備份檔)
ansible clients -i ansible/inventory.ini \
  -m ansible.posix.synchronize \
  -a "mode=pull src=/home/pi/bestweights/ dest=bestweights/pi_bestweights/ rsync_opts='--exclude=periodic/'"

# 從 VPS 拉回 Server 端結果
## 拉回 Server 端日誌
# (注意：如果日誌太大，建議只拉回需要的 CSV)
rsync -avz --exclude='periodic/' ubuntu@51.254.207.168:~/csc8114/code/results/ results/vps_results/

## 拉回 Server 端權重 (排除 periodic 備份檔以節省空間)
rsync -avz --exclude='periodic/' ubuntu@51.254.207.168:~/csc8114/code/bestweights/ bestweights/


# 資料整理與報表生成流程 (Post-Processing)
---
## 9. Data Organization & Merging Pipeline / 數據整理與合併流程
拉回資料後，請依照下列順序進行預處理，確保報表能夠正確生成：

### 9.1 整理日誌 (Logs / CSV)
> [!TIP]
> **預覽執行結果 (Dry Run)**：若要在正式搬移前確認邏輯，可加入 `--dry-run`。
> 例如：`python3 scripts/organize_by_seed.py --dry-run` 或 `python3 scripts/flatten_for_report.py --dry-run --move`

```bash
# 1. 刪除進度暫存檔 (只保留正式日誌)
python3 scripts/cleanup_progress.py

# 2. 依照 Seed 分類日誌 (建立 seed42, seed52 等子目錄)
python3 scripts/organize_by_seed.py

# 3. 統一資料夾命名格式 (例如 seed42 -> 01_seed42)
python3 scripts/rename_seed_dirs.py

# 4. 打平目錄並合併到主 results 資料夾 (使用 --move 節省空間)
python3 scripts/flatten_for_report.py --move

# 5. 合併 VPS 的日誌到主目錄 (若有需要)
mv -n results/vps_results/2026-04-09_08-11-48/* results/2026-04-09_08-11-48/
```

### 9.2 整理權重 (Weights / PTH)
> [!TIP]
> **權重分類預覽**：若不確定分類結果，可執行 `.venv/bin/python scripts/classify_client_weights.py --dry-run`

```bash
# 1. 讀取 .pth 內容並精準分類 Client 權重 (需使用虛擬環境)
.venv/bin/python scripts/classify_client_weights.py

# 2. 將 Client 權重合併至 Server 權重目錄
# (這一步讓 eval-batch 能夠在同一個資料夾找到 head 和 base)
for seed_dir in bestweights/pi_bestweights/2026-04-09_08-11-48/*/; do
    name=$(basename "$seed_dir")
    dst="bestweights/2026-04-09_08-11-48/$name"
    if [ -d "$dst" ]; then
        mv "$seed_dir"best_client_*.pth "$dst/" 2>/dev/null
    fi
done

# 3. 清理空資料夾 (選放，節省空間)
rm -rf bestweights/pi_bestweights
```

## 10. Matrix Reporting Pipeline / 矩阵报表生成流

### 10.1 指令功能與需求說明 (Roles & Requirements)
*   **`make eval-batch` (計算者 / Calculator)**:
    *   **任務**: 執行模型推理，計算 MSE, F1, Accuracy 等準確度指標。
    *   **所需檔案**: 
        *   **權重 (Weights)**: `server_head_*.pth` 與 `best_client_*.pth` (需合併至同一目錄)。
        *   **數據 (Dataset)**: `/data/processed/` 下的 `.parquet` 檔案。
    *   **產出**: 每個場景獨立的 **`.json`** 檔案 (存於 `reports/`)，封存原始指標。
*   **`make matrix-report` (彙整者 / Secretary)**:
    *   **任務**: 彙整所有場景、Seed 的數據，產出最終比較表格。
    *   **所需檔案**: 
        *   **準確度指標**: `reports/*.json`。
        *   **效率指標**: `results/SESSION/` 下的 `training_log_client*.csv` (提取延遲與流量)。
    *   **產出**: 
        *   **`Matrix_Global_Summary_*.csv`**: **論文核心表格**，彙整所有場景平均效能。
        *   **`Matrix_Station_Details_*.csv`**: 詳細的場景 x 月份數據。

### 10.2 執行指令
```bash
# 1. 執行批次評估 (計算準確度指標)
make eval-batch SESSION=2026-04-09_08-11-48 2>&1 | tee reports/evaluation_console_log.txt

# 2. 執行報表彙整 (提取準確度 + 延遲 + 流量)
make matrix-report SESSION=2026-04-09_08-11-48 2>&1 | tee reports/matrix_report_console.log
```
