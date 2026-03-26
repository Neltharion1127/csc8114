# CSC8114 Group 9 — Joint Computation and Communication Optimisation in Federated Split Learning for IoT Rainfall Prediction

> A production-grade Federated Split Learning system with adaptive communication control for real-time IoT rainfall prediction.

## Overview

In real-world IoT sensing networks, distributed weather stations continuously generate large volumes of environmental data. Traditional centralised training requires raw data to be transmitted to a central server — introducing **privacy risks**, **high bandwidth costs**, and **single points of failure**. Federated Split Learning (FSL) addresses this by splitting the neural network between edge clients and a central server: clients compute only the early layers locally and transmit compact intermediate activations instead of raw data.

However, FSL introduces its own communication bottleneck — **per-step activation/gradient round trips** and **periodic model synchronisation** can dominate training time under constrained or unstable IoT networks. This project tackles this problem head-on.

### What We Built

A fully functional, end-to-end FSL training platform built on a **gRPC microservice architecture** with the following core components:

- **Split LSTM Model** — A 2-layer LSTM encoder runs on each client to extract temporal weather features from a 24-hour sliding window (5 meteorological variables), producing a fixed-size 64-dim _smashed activation_. The server hosts a dual-head prediction module (rain/no-rain classifier + rainfall amount regressor) with LayerNorm, SiLU activations, and focal loss to handle severe class imbalance.

- **gRPC Communication Layer** — Four well-defined RPCs (`Register`, `Forward`, `Synchronize`, `NotifyCompletion`) orchestrate the full distributed training lifecycle, including client registration, split forward/backward passes, barrier-based FedAvg aggregation (with configurable quorum and timeout), and graceful shutdown.

- **Adaptive Communication Scheduler** — A latency-aware, rule-based scheduler running on the server side that jointly controls two optimisation dimensions at runtime:
  - **Spatial** — Activation compression mode (float32 → float16 → int8 → top-k sparsification), reducing per-step payload by up to **73%**.
  - **Temporal** — Synchronisation interval ρ (1–20 local epochs between FedAvg rounds), reducing server aggregation frequency by up to **77%**.
  
  Decisions are driven by EMA-smoothed client-reported latency, enabling the system to dynamically adapt to heterogeneous and non-stationary network conditions without manual tuning.

- **Reproducible Experiment Matrix** — 14 carefully designed scenarios × 10 random seeds, systematically decoupling the effects of latency profiles (none / light / medium / burst / extreme), compression modes, synchronisation intervals, and adaptive vs. static policies. All experiments are orchestrated via a single `config.yaml` and `make matrix` command.

### Key Results

| Finding | Detail |
| ------- | ------ |
| INT8 compression | **−73.4% payload** (256 B → 68 B) with only −0.037 Macro F1 |
| ρ = 3 sync interval | **−54.6% runtime**, server rounds 30 → 10, only −0.016 Macro F1 |
| Adaptive under burst latency | **−20.7% payload** with near-zero F1 impact |
| Adaptive under extreme latency | **−63.4% runtime** with moderate F1 trade-off |

No single strategy dominates — practical deployment should choose policies based on bandwidth budget, latency stability, and acceptable accuracy degradation.

---

## Repository Structure

```
csc8114/
├── code/                        # FSL system implementation
│   ├── config.yaml              # All training, model, and experiment configs
│   ├── Makefile                 # Build / run / plot automation
│   ├── proto/fsl.proto          # gRPC protocol (4 RPCs)
│   ├── src/
│   │   ├── models/              # ClientLSTM (encoder) + ServerHead (predictor)
│   │   ├── nodes/               # Server and client entry points
│   │   ├── client/              # Training loop, forward step, sync, checkpointing
│   │   ├── server/              # Forward service, FedAvg, adaptive scheduler
│   │   ├── shared/              # Compression, serialisation, config, runtime
│   │   └── data/                # Data download, evaluation, plotting
│   ├── dataset/                 # Raw and processed data
│   ├── bestweights/             # Model checkpoints (per session)
│   └── results/                 # Training logs, evaluation reports, plots
│
└── paper/                       # IEEE conference paper (IEEEtran)
    ├── csc8114 assessment1.tex  
    ├── csc8114.tex              # Main aper
    ├── refs.bib                 # Bibliography
    └── diagrams/                # Architecture & sequence diagrams (Mermaid + PNG)
```

---

## Quick Start

### Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- (Optional) Docker & Docker Compose for containerised runs
- (Optional) TeX Live for paper compilation

### Environment Setup

```bash
cd code/
uv sync
```

### Data Download

```bash
make download-data
```

Downloads and preprocesses Open Meteo historical weather data for 12 Newcastle stations.

### Training (Native — Recommended)

```bash
make clean-native
make run-native
```

This starts one server + 3 client processes locally with colour-prefixed logs, and auto-plots results on completion.

Override devices or address:

```bash
make run-native SERVER_HOST=127.0.0.1 SERVER_DEVICE=cpu CLIENT_DEVICE=mps
```

### Training (Docker)

```bash
make run-network    # start
make clean          # teardown
```

### Evaluation & Plotting

```bash
make eval-latest                              # evaluate latest session
make eval-session SESSION=<session-id>        # evaluate specific session
make plot-latest                              # plot latest session
make plot-session SESSION=<session-id>        # plot specific session
```

### Experiment Matrix

```bash
make matrix-dry-run                # preview scenario plan
make matrix                        # run all 14 scenarios
make matrix ONLY=M01,M10 MAX_RUNS=2  # run selected scenarios
```

Run `make help` for the full command list.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      gRPC Server                        │
│  ┌────────────┐  ┌───────────┐  ┌────────────────────┐  │
│  │ ServerHead │  │  FedAvg   │  │ Adaptive Scheduler │  │
│  │ (MLP+Dual  │  │ Coordinator│  │ (Compression + ρ)  │  │
│  │  Head)     │  │           │  │                    │  │
│  └────────────┘  └───────────┘  └────────────────────┘  │
└──────────┬──────────────┬──────────────┬────────────────┘
           │ Forward/     │ Synchronize  │ Register/
           │ Backward     │ (FedAvg)     │ Complete
           │              │              │
┌──────────▼──────────────▼──────────────▼────────────────┐
│              gRPC Clients (×3)                          │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ClientLSTM│  │Training Loop │  │  Compression &   │  │
│  │(2L LSTM) │  │(local steps) │  │  Latency Report  │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Model (Split LSTM)

| Component      | Location                   | Description                                                                 |
| -------------- | -------------------------- | --------------------------------------------------------------------------- |
| **ClientLSTM** | `src/models/split_lstm.py` | 2-layer LSTM encoder; input (batch, 24, 5) → smashed activation (batch, 64) |
| **ServerHead** | `src/models/split_lstm.py` | MLP backbone → dual head: rain classifier (logit) + rain regressor (amount) |

### gRPC Protocol (`proto/fsl.proto`)

| RPC                | Direction       | Purpose                                                             |
| ------------------ | --------------- | ------------------------------------------------------------------- |
| `Register`         | Client → Server | Register client, receive assigned ID and session                    |
| `Forward`          | Client ↔ Server | Send compressed activation, receive gradient + scheduler directives |
| `Synchronize`      | Client ↔ Server | FedAvg weight aggregation (barrier-based with quorum + timeout)     |
| `NotifyCompletion` | Client → Server | Signal training complete; triggers shutdown when all clients finish |

### Adaptive Scheduler (`src/server/scheduler.py`)

Uses EMA-smoothed latency to escalate compression and synchronisation interval:

| Latency (EMA) | Compression Mode  | ρ Increment |
| ------------- | ----------------- | ----------- |
| < 4 ms        | float32 (default) | +0          |
| 4–10 ms       | float16           | +1          |
| 10–15 ms      | int8              | +2          |
| > 15 ms       | top-k             | +3          |

### Compression Modes (`src/shared/compression.py`)

| Mode    | Encoding                         | 64-dim Payload   |
| ------- | -------------------------------- | ---------------- |
| float32 | Raw numpy                        | 256 B            |
| float16 | Half-precision                   | 128 B            |
| int8    | Scale (4B) + quantised           | 68 B             |
| top-k   | Header + sparse indices + values | Depends on ratio |

---

## Configuration (`config.yaml`)

Key configuration sections:

| Section               | Description                                                         |
| --------------------- | ------------------------------------------------------------------- |
| `model.*`             | LSTM hidden size, layers, sequence length, horizon                  |
| `training.*`          | Learning rate, rounds, local steps, loss weights, focal loss params |
| `compression.*`       | Default mode, top-k ratio                                           |
| `federated.*`         | Number of clients, base ρ                                           |
| `scheduler.*`         | Latency thresholds, ρ bounds, EMA alpha                             |
| `profiler.*`          | Latency generator settings (base, offsets, jitter, burst)           |
| `experiment_matrix.*` | 14 scenario definitions with per-scenario overrides                 |

---

## Experiment Matrix (M01–M14)

| Profile    | ID      | Compression                       | ρ               | Scheduler |
| ---------- | ------- | --------------------------------- | --------------- | --------- |
| No Latency | M01     | Float32 (fixed)                   | 1               | Disabled  |
| No Latency | M02     | Starts Float32                    | Dynamic         | Enabled   |
| Light      | M03–M04 | Float32 / Adaptive                | 1 / Dynamic     | Off / On  |
| Medium     | M05–M10 | Float32 / INT8 / Top-k / Adaptive | 1 / 3 / Dynamic | Various   |
| Burst      | M11–M12 | Float32 / Adaptive                | 1 / Dynamic     | Off / On  |
| Extreme    | M13–M14 | INT8 / Adaptive                   | 1 / Dynamic     | Off / On  |

---

## Output Structure

Each training session produces:

```
bestweights/<session>/           # Model checkpoints
  ├── server_head_round_*.pth    # Latest server checkpoint
  └── periodic/                  # Periodic paired checkpoints

results/<session>/               # Logs, metrics, plots
  ├── training_log_client*.csv   # Per-client training/validation logs
  ├── server_log_*.csv           # Server-side aggregation logs
  ├── evaluation_report_*.json   # Final test metrics (source of truth)
  ├── training_curve_*.png       # Training curves
  ├── server_metrics_*.png       # Server metrics
  └── confusion_matrix_*.png     # Confusion matrices
```

---

## Paper

The IEEE conference paper is in `paper/`. To compile:

```bash
cd paper/
pdflatex "csc8114 accessment1.tex"
bibtex "csc8114 accessment1"
pdflatex "csc8114 accessment1.tex"
pdflatex "csc8114 accessment1.tex"
```

> Run `pdflatex` three times to resolve all cross-references and citations.

Diagrams are authored in [Mermaid](https://mermaid.js.org/) and exported via [mermaid.live](https://mermaid.live/). Always edit and export from the website for consistency.

---

## Team

| Member        | Role                                 |
| ------------- | ------------------------------------ |
| Baoyi Liu     | Related Work & Writing               |
| Guanghua Liu  | Related Work & Writing               |
| Jiale Liu     | Introduction & Literature Review     |
| Mingyuan Shao | Compression & Discussion             |
| Wenjie Ding   | Methodology, System Design & Results |
| Yi Sin Lin    | System Design & Implementation       |
| Zhuolun Li    | Experiments & Conclusion             |

School of Computing, Newcastle University