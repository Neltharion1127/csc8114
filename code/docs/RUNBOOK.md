# FSL Experiment Runbook / 实验运行手册

---

## Prerequisites / 前置条件

| Requirement / 条件 | Check / 检查方式 |
|---|---|
| Tailscale connected / Tailscale 已连接 | `tailscale status` |
| Docker Hub logged in / Docker Hub 已登录 | `docker login` |
| Ansible installed / Ansible 已安装 | `ansible --version` |
| VPS reachable / VPS 可达 | `ping 51.254.207.168` |

---

## 1. Build & Push Image / 构建并推送镜像

Only needed when `src/`, `Dockerfile`, `pyproject.toml`, or `proto/` change.
仅在修改了 `src/`、`Dockerfile`、`pyproject.toml` 或 `proto/` 时需要执行。

```bash
git add .
git commit -m "..."
git push origin main
```

GitHub Actions builds a multi-arch image (amd64 + arm64) and pushes it to Docker Hub automatically. Wait ~5–10 minutes before proceeding.

GitHub Actions 自动构建 amd64+arm64 双架构镜像并推送到 Docker Hub，等待约 5-10 分钟后再继续。

---

## 2. Start VPS Server / 启动 VPS 服务端

**First time or after changing config.yaml / 第一次或修改 config.yaml 后：**

```bash
scp code/config.yaml ubuntu@51.254.207.168:~/config.yaml
scp code/docker-compose.server.yml ubuntu@51.254.207.168:~/docker-compose.server.yml
```

**Every time / 每次启动：**

```bash
ssh ubuntu@51.254.207.168

docker pull cindyncl26/fsl-client:latest
docker compose -f docker-compose.server.yml down 2>/dev/null
docker compose -f docker-compose.server.yml up -d
docker logs -f fsl-server
# Wait for: "Listening for incoming FSL connections on [::]:50051"
# 等待看到: "Listening for incoming FSL connections on [::]:50051"
```

---

## 3. Deploy Clients to All Pis / 部署客户端到所有 Pi

Run from the `code/` directory on your local machine.
在本机 `code/` 目录下执行。

```bash
cd code
ansible-playbook ansible/deploy_client.yml -i ansible/inventory.taisale.ini
```

This single command will automatically:
这一条命令会自动完成：

1. Create required directories on all Pis / 在所有 Pi 上创建必要目录
2. Sync `dataset/processed/` (incremental, slow on first run) / 同步数据集（增量，第一次较慢）
3. Sync latest `config.yaml` to every Pi / 同步最新 `config.yaml` 到每台 Pi
4. Pull latest image / 拉取最新镜像
5. Stop old container and start new training / 停掉旧容器，启动新容器开始训练

---

## 4. Monitor Progress / 查看进度

**VPS server logs / VPS 服务端日志：**
```bash
ssh ubuntu@51.254.207.168 "docker logs -f fsl-server"
```

**Single Pi client log / 单台 Pi 客户端日志：**
```bash
ssh -i ~/.ssh/id_rsa pi@100.125.109.94 "docker logs -f fsl-client"
```

**All Pi container status / 所有 Pi 容器状态：**
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -a "docker ps --filter name=fsl-client --format '{{.Names}}: {{.Status}}'"
```

---

## 5. Change Config & Re-run / 改参数重跑

```
1. Edit code/config.yaml locally / 修改本地 code/config.yaml
2. scp config.yaml to VPS (step 2) / 重新推 config 到 VPS（步骤二）
3. Restart VPS server (step 2) / 重启 VPS 服务端（步骤二）
4. Re-deploy clients (step 3) / 重新部署客户端（步骤三）
```

No need to push code or rebuild the image.
不需要 push 代码，不触发 CI 重新构建镜像。

---

## 6. Stop All Training / 停止所有训练

**Stop all Pi clients / 停止所有 Pi 客户端：**
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -a "docker stop fsl-client" \
  --become
```

**Stop VPS server / 停止 VPS 服务端：**
```bash
ssh ubuntu@51.254.207.168 "docker stop fsl-server"
```

---

## 7. Collect Results / 收集实验结果

**Pull results from all Pis / 从所有 Pi 拉取结果：**
```bash
ansible clients -i ansible/inventory.taisale.ini \
  -m ansible.posix.synchronize \
  -a "mode=pull src=/home/pi/results/ dest=code/results/pi_results/"
```

**Pull results from VPS / 从 VPS 拉取结果：**
```bash
scp -r ubuntu@51.254.207.168:~/results/ code/results/vps_results/
```

---

## 8. Manual Debug on Single Pi / 单台 Pi 手动调试

```bash
ssh -i ~/.ssh/id_rsa pi@100.125.109.94

CLIENT_ID=1 docker compose -f docker-compose.client.yml up
```

---

## Quick Reference / 快速参考

| Action / 操作 | Command / 命令 |
|---|---|
| Trigger image build / 触发镜像构建 | `git push origin main` |
| Start server / 启动服务端 | VPS: `docker compose -f docker-compose.server.yml up -d` |
| Deploy all clients / 部署所有客户端 | `ansible-playbook ansible/deploy_client.yml -i ansible/inventory.taisale.ini` |
| Watch server logs / 看服务端日志 | `docker logs -f fsl-server` |
| Stop all clients / 停止所有客户端 | `ansible clients -i ... -a "docker stop fsl-client" --become` |
| Deploy to subset of Pis / 只部署部分 Pi | append `--limit pi01,pi02` to ansible command |
| Run specific matrix scenarios / 跑特定场景 | `python -m src.data.run_experiment_matrix --only M06,M07` |
