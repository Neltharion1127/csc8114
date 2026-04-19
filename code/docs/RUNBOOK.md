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
| 全新启动实验 | `make dist-start` |
| 核弹重启（停止+清空+重启）| `make dist-restart` |
| 只更新镜像到 Pi（Mac 构建）| `make dist-load-image-local` |
| 只更新镜像到 Pi（VPS 中转）| `make dist-load-image` |
| 同步 config 到双端 | `make dist-sync-config` |
| 只重启服务端 | `make dist-server-restart` |
| 查看服务端日志 | `make dist-logs` |
| 清空 VPS 结果 | `make dist-clean-server` |
| 清空 Pi 结果 | `make dist-clean-results` |
| 构建并推送镜像到 Docker Hub | `make dist-build` |
| 只跑指定场景 | `make matrix ONLY=05,06` |

---

## 10. Multi-Scenario Ablation Study (Automated) / 全自動消融實驗矩陣流程

此流程專為連續運行 14 個消融場景所設計，具備 Metadata 標頭檢查與自動故障恢復能力。

### Step 1: Nuclear Cleanup / 核彈級深度清理
啟動前必須先清理所有機器，避免「幽靈程序」干擾同步：

```bash
# A. Mac 執行：殺死所有 Pi 上的殘留 Python 程序與容器，並清空 Pi 的結果與權重 (看到 FAILED 為正常)
ansible clients -i ansible/inventory.ini -m shell \
  -a "docker stop fsl-client; docker rm fsl-client; pkill -9 python; pkill -9 python3; rm -rf /home/pi/results/* /home/pi/bestweights/*" --become

# B. Mac 執行：刪除 Pi 上未使用的舊鏡像 (釋放 1.6GB+ 磁碟空間)
ansible clients -i ansible/inventory.ini -m shell \
  -a "docker image prune -a -f" --become

# C. VPS 執行：重設伺服器與「徹底清空」舊權重與結果
docker compose -f docker-compose.server.yml down
rm -rf ~/csc8114/code/bestweights/*
rm -rf ~/csc8114/code/results/*
```

### Step 2: Build, Sync & Update / 建置與分發
```bash
# A. Mac 執行：重新封裝代碼 (Metadata 保險機制) 並推送
make dist-build

# B. Mac 執行：同步 matrix.yaml 配置
make dist-sync-config

# C. Mac 執行：讓樹莓派拉取最新鏡像
ansible-playbook ansible/deploy_client.yml -i ansible/inventory.ini --tags "pull"

# D. VPS 執行：讓伺服器拉取最新鏡像
docker compose -f docker-compose.server.yml pull
```

### Step 3: Launch / 啟動全自動循環
建議先啟動 Server，等待約 5 秒後再從 Mac 啟動 Client：

```bash
# 1. VPS 啟動 Server
docker compose -f docker-compose.server.yml up -d

# 2. Mac 啟動 Client 循環
ansible-playbook ansible/deploy_client.yml -i ansible/inventory.ini --tags "run"
```
