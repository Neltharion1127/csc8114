# 联邦拆分学习 (FSL) 零基础复现学习计划

我为你整理了一个由浅入深的**五阶段学习与复现计划**。建议你按照这个顺序，一个模块一个模块地写，每写完一个阶段就做一次简单的测试。

---

## 阶段一：夯实基础（环境与数据）

**目标**：搭建好房屋的地基，确保能顺利拿到训练数据。

1. **初始化项目与环境**
   - 创建 `pyproject.toml`，使用 `uv` 初始化 Python 3.11+ 环境。
   - 熟悉配置驱动开发：创建一个 `config.yaml`，把你所有的超参数（如客户端数量、LSTM隐藏层大小、学习率等）都写进去，保证后面的代码不出现“硬编码（Hardcode）”。
   - *学习点：如何使用声明式包管理工具和 YAML 配置解析。*
   
2. **数据拉取与预处理 (`data/fetch_uo.py`)**
   - 编写脚本调用 Urban Observatory API `/sensors/data/json`，拉取 Rainfall（降雨）等数据。
   - 使用 `pandas` 处理时间戳、填充缺失值（Mock 湿度/风速），最后存为离线 CSV。
   - *学习点：处理真实世界不规则的时序数据 API。*

3. **构建 Pytorch 数据集 (`data/dataloader.py`)**
   - 编写 `RainfallDataset` 类（继承 `torch.utils.data.Dataset`），实现时序数据的“滑动窗口（Sliding Window）”切片（比如用过去24小时预测未来1小时）。
   - 实现**数据划分逻辑**：按传感器地理位置把数据分给 Client 0, 1, 2，确保他们拿到的是 **Non-IID（非独立同分布）** 的数据。
   - 分割 Train/Val/Test 并在 Train 上去做 StandardScaler 归一化。
   - *测试方法：写个简单的脚本迭代 DataLoader，打印出 X 和 Y 的 `shape`。*

---

## 阶段二：神经网络的核心（模型与切分）

**目标**：理解拆分学习（Split Learning）是如何切开一个完整神经网络的。

1. **定义客户端模型 (`ClientLSTM`)**
   - 使用 `torch.nn.LSTM`。输入为特征维度，输出为隐层状态（Hidden State）。
   - 这个网络的输出就是将被传输的“激活值（Smashed Activation）”。
   
2. **定义服务端模型 (`ServerHead`)**
   - 使用 `torch.nn.Linear` 预测头，输入是接收到的激活值的维度，输出是 1 维（预测的降雨量）。
   
3. **理解前向与反向传播的断点连接**
   - *思考题*：如果 Client 输出了一个张量 $A$，传给 Server 变成了 $A'$，Server 求 Loss 并执行 `loss.backward()` 后，怎么把 $A'$ 的梯度传回给 Client，让 Client 的 $A$ 继续 `A.backward(grad)`？
   - *测试方法*：写一个 mock 脚本，实例化这两个类，假装把 Client 的输出喂给 Server，看能不能走通一次完整的梯度更新。

---

## 阶段三：搭建通信桥梁（gRPC 与序列化）

**目标**：掌握真实分布式系统中最常用的通信协议，并实现张量的网络传输。

1. **定义协议接口 (`proto/fsl.proto`)**
   - 学习 Protobuf 语法。定义 `ForwardRequest`（包含客户端的二进制激活值、当前压塑模式、延迟、以及真实 Target 标签）和 `ForwardResponse`（包含服务端传回来的二进制梯度）。
   - 定义 `SyncRequest` 和 `SyncResponse` 用于传输模型权重。
   - 运行 `python -m grpc_tools.protoc` 编译出 Python 代码。
   
2. **张量序列化工具 (`shared/serialization.py`)**
   - 神经网络的 Tensor 不能直接在网上传。你需要写工具函数把 PyTorch Tensor 保存为内存中的 bytes (`io.BytesIO()`)。
   - 同样需要写工具把 `model.state_dict()` 转成二进制流。
   - *测试方法：把一个随机 Tensor 转成 bytes，再转回来，用 `torch.allclose` 判断是否完全一致。*

---

## 阶段四：联邦与压缩核心逻辑（客户端与服务端）

**目标**：真正把你的 FSL 系统串联起来，实现论文的两大创新点。

1. **实现客户端逻辑 (`client/client.py`)**
   - **构建主循环**：取一个 Batch -> 本地 Forward -> **调用压缩模块 (`shared/compression.py`) 将浮点数转为 float16/int8 代价极小的 bytes** -> 通过 gRPC stubs 发送给 Server -> 收到梯度 -> 本地 Backward。
   - **Profiler**：在每一次 Forward 前后记录 `time.time()` 算出 Latency，记录 Payload 的 `len(bytes)` 算出上传量。
   - *学习点：如何将前向传播与网络 I/O 深度融合。*

2. **实现服务端协调器 (`server/coordinator.py`)**
   - 继承 gRPC 生成的 `Servicer` 类。
   - 在 `Forward` 方法里：解压数据 -> 获取目标标签 -> Server 本地 Forward -> 算 Loss -> Backward -> 提取梯度 -> 序列化返回。
   - **调度器 (`server/scheduler.py`) 引入**：根据客户端在 request 里报上来的 Latency，用一条 If/Else 规则判断是否要让他下一次改用更狠的压缩算法（比如 int8）。
   
3. **实现联邦平均 (`server/sync_manager.py`)**
   - 每隔 $\rho$ 个 Round，客户端调用 `Synchronize` 接口把当前模型权重发给服务端。
   - 服务端必须维护一个线程锁（`threading.Lock`）和事件障碍（`threading.Event`），阻塞先到的客户端，直到等齐所有客户端的权重后，做几何平均相加（FedAvg），再统一放行返回全局权重。
   - *学习点：多线程编程中的锁机制与状态同步。*

---

## 阶段五：运行、测试与可视化

**目标**：在本地跑起来，并为你的论文产出图表。

1. **编写单机联邦仿真脚本 (`sim/run_local.py`)**
   - 使用 Python 原生的 `multiprocessing` 库。写一个脚本在后台启动 Server 进程，然后在一转头拉起 3 个 Client 进程。
   - 观察终端里它们交互的日志，跑通你的第一个 Epoch。
   - *学习点：进程管理与并发程序调试。*
   
2. **实验数据记录 (`server/metrics.py`)**
   - 让 Server 在每次 Forward 后，把当前 Round、压缩模式、rho、Latency、Loss、上传字节数写到一个 CSV 文件里。
   
3. **数据可视化 (`utils/plotting.py`)**
   - 使用 `pandas` 读取生成的 CSV，利用 `matplotlib` / `seaborn` 绘制 折线图（RQ2: 不同 $\rho$ 的收敛速度对比）和散点柱状双轴图（RQ1: 压缩精度与通信字节的 Trade-off）。

4. **进阶挑战：容器化部署 (Docker)**
   - 编写 `Dockerfile.client` 和 `Dockerfile.server`。
   - 编写 `docker-compose.yml` 把它们放到同一个虚拟网络里。这一步能证明你的系统是真正的“分布式”，而不仅仅是单机多进程。

---

祝你重写顺利！每遇到一个坎，都可以把我之前的代码拿出来对照一下（如果你已经删了，我也能随时教你那一步具体怎么写）。享受从 0 到 1 创造系统的过程吧！
