# 联邦拆分学习 (FSL) 系统架构与逻辑说明

本文档详细说明了本项目针对降雨预测任务实现的**联邦拆分学习（Federated Split Learning, FSL）**系统的整体逻辑和模块架构。

---

## 1. 核心思想与系统目标

本系统结合了**联邦学习（Federated Learning）**保护数据隐私的特性与**拆分学习（Split Learning）**降低客户端计算压力的优势。系统部署在模拟的物联网（IoT）传感器节点（客户端）和云端服务器上。

为了解决真实 IoT 场景下的通信瓶颈问题，系统引入了以下核心机制：
1. **空间压缩 (Spatial Compression)**：通过激活值量化（float32 -> float16/int8）和 Top-k 稀疏化，减少单词前向/反向传播的数据传输量。
2. **时间解耦 (Temporal Decoupling)**：引入可调的全局同步频率 $\rho$，不必每个 batch 都进行全局权重同步，从而大幅节省通信开销。
3. **自适应调度 (Adaptive Scheduling)**：服务端根据客户端实时上报的网络延迟指标，动态决定下一个 round 的压缩级别和同步频率。

---

## 2. 整体工作流程 (Workflow)

系统的训练过程高度依赖基于 **gRPC** 的通信流，分为**前向传播、反向传播**和**全局聚合**三个阶段。

### A. 前向与反向传播 (每 1 个 Step 发生)
1. **本地特征提取**：Client 取出一个 batch 的本地降雨数据，仅通过 `ClientLSTM` 网络计算，输出中间隐层特征（Smashed Activation）。
2. **激活值压缩**：Client 调用 `ActivationCompressionModule`，根据当前配置的 `compression_mode` 将激活值压缩为二进制流。同时附带本地测量的 `latency`（延迟）、`bytes_uploaded`（已传字节）和真实的 `target`（标签）。
3. **RPC 传输 (Forward)**：Client 通过 gRPC 将压缩后的 Payload 打包进 `ForwardRequest` 发送给 Server。
4. **服务端计算**：
   - Server 收到请求后，先利用 `Scheduler` 根据 Client 的 `latency` 动态更新该 Client 接下来的压缩配置和 $\rho$。
   - 解压激活值，输入到 `ServerHead` 完成前向计算，得出预测结果。
   - 使用真实的 `target` 计算 MSE 损失（Loss），并进行反向传播计算梯度。
5. **梯度回传**：Server 提取截断层（Cut Layer）的梯度，将其打包进 `ForwardResponse`，连同新的压缩和调度配置一起返回给 Client。
6. **本地反压**：Client 接收到梯度，完成本地 `ClientLSTM` 的反向传播并使用优化器更新本地权重。

### B. 全局权重同步 (每 $\rho$ 个 Rounds 发生一次)
1. **触发同步**：当 Client 训练到第 $\rho$ 轮的整数倍时，暂停本地训练。
2. **RPC 传输 (Synchronize)**：Client 提取本地 `ClientLSTM` 的模型权重（State Dict），通过 `SyncRequest` 发送给 Server。
3. **联邦平均 (FedAvg)**：Server 端的 `SynchronizationManager` 阻塞等待，直到收集齐所有 Client 的权重。然后执行 **FedAvg**（对所有权重求几何平均数），得到全局平均权重。
4. **权重下发**：Server 将全局平均权重包装在 `SyncResponse` 中返回给各个 Client。Client 加载全局权重，进入下一个 $\rho$ 周期的训练。

---

## 3. 代码目录与模块职责

所有核心代码均位于 `code/` 目录下，结构按职责严格划分：

### 📌 基础配置与通信层
*   **`config.yaml`**: 系统的全局配置文件，定义了模型结构参数、训练超参、初始压缩模式、和同步频率参数。
*   **`proto/fsl.proto`**: 定义了 Client 和 Server 交互的数据结构 (Messages) 和远程过程调用接口 (RPCs)。
*   **`shared/serialization.py`**: 提供高效的方法将 PyTorch 的张量（Tensors）和模型参数（Weights）转化为最少体积的二进制 Bytes 以便网络传输。
*   **`shared/compression.py`**: 核心的空间压缩模块。包含了常规精度(`float32`)，半精度(`float16`)，8位量化(`int8`)和基于比例的稀疏化(`topk`)算法。

### 📌 数据处理层
*   **`data/fetch_uo.py`**: 用于从 Urban Observatory API 拉取真实的纽卡斯尔降雨数据，并清洗保存为 CSV 格式。
*   **`data/dataloader.py`**: 负责将离线 CSV 数据转换为 PyTorch Dataset。支持滑动窗口（Sliding Window）切片，并将数据集按地理空间非独立同分布（Non-IID）地分割给不同的 Client。

### 📌 模型与算法层
*   **`models/split_lstm.py`**: 定义了骨干网络结构。包含客户端的 `ClientLSTM` 网络和服务端的带预测头的 `ServerHead` 网络。
*   **`client/client.py` & `client/profiler.py`**: FSL 客户端主程序。管理着本地的数据加载器、前向后向计算逻辑以及与 Server 的 gRPC 通信。`ClientProfiler` 负责监控客户端的网络传输性能。
*   **`server/coordinator.py`**: FSL Server 端入口。作为 gRPC 服务端接收请求，处理服务端的梯度反压和逻辑分发。
*   **`server/scheduler.py`**: 规则调度器（Rule-based Scheduler）。监控收集上来的延迟数据，一旦延迟超过阈值，自动升级该客户端的压缩级别或降低同步频率。
*   **`server/sync_manager.py`**: 线程安全的权重聚合管理器，负责执行 FedAvg 算法。
*   **`server/metrics.py`**: 实时将每个 Client 每轮的 Loss、延迟和传输字节数落盘保存为 CSV 文件，供后续作图分析使用。

### 📌 实验与运行层
*   **`experiments/run_sweep.py`**: 用于自动扫描不同 `mode` (压缩配置) 和 `rho` (同步频率) 组合的自动化实验脚本。
*   **`sim/run_local.py`**: (仿真器) 由于在个人电脑上开启多个 Docker 并发可能存在资源和守护进程限制，该脚本利用 Python 多线程原生模拟 3 客户端 + 1 服务端的真实交互架构，方便快速 Debug。
*   **`utils/plotting.py`**: 读取 metrics 数据，并生成精美的论文插图 (解答论文中 RQ1, RQ2, RQ3 的图表)。
*   **`docker-compose.yml` & `docker/`**: 容器化部署方案，保障在跨机器或边缘设备（如树莓派）上能有一致的运行环境。

---

## 4. 如何查阅和运行实验

**启动单次系统仿真测试：**
```bash
uv run python sim/run_local.py --mode float32 --rho 1
```

**运行一键自动化参数调优实验：**
```bash
uv run python experiments/run_sweep.py
```

**生成可视化图表：**
```bash
uv run python utils/plotting.py
```
*(生成的图表保存在 `code/outputs/plots` 中，评估数据保存在 `code/outputs/metrics` 中)*
