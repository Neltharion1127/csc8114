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

- **Reproducible Experiment Matrix** — 14 carefully designed scenarios × 3 random seeds, systematically decoupling the effects of latency profiles (none / mid / high), compression modes, synchronisation intervals, and adaptive vs. static policies. Scenarios are defined in `matrix.yaml` and run with a single `make matrix` command; `config.yaml` remains the clean single-run baseline.

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
│   ├── config.yaml              # Runtime config — model, training, data, comms
│   ├── matrix.yaml              # Experiment matrix — 14 scenarios & seeds (ablation only)
│   ├── Makefile                 # Build / run / plot automation
│   ├── proto/fsl.proto          # gRPC protocol (4 RPCs)
│   ├── src/
│   │   ├── models/              # ClientLSTM (encoder) + ServerHead (predictor)
│   │   ├── nodes/               # Server and client entry points
│   │   ├── client/              # Training loop, forward step, sync, checkpointing
│   │   ├── server/              # Forward service, FedAvg, adaptive scheduler
│   │   ├── shared/              # Compression, serialisation, config, runtime
│   │   └── data/                # Data download, evaluation, plotting
│   ├── dataset/
│   │   ├── processed/           # 11 training sensor files (one per federated client)
│   │   └── holdout/             # 1 holdout sensor (NCL_GATESHEAD — never seen during training)
│   ├── bestweights/             # Model checkpoints (per session)
│   └── results/                 # Training logs, evaluation reports, plots
│
└── paper/                       # IEEE conference paper (IEEEtran)
    ├── csc8114 assessment1.tex  
    ├── csc8114.tex              # Main paper
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

Downloads Open-Meteo historical weather data (2023–2026) for 12 Newcastle stations:

- **11 training locations** → `dataset/processed/` (one file per federated client, north bank of the Tyne)
- **1 holdout location** → `dataset/holdout/` (NCL_GATESHEAD — south bank, geographically isolated, never seen during training)

### Training (Native — Recommended for development)

```bash
make native-clean
make native-run SCENARIO_ID=01                          # single scenario, all 11 clients
make native-run SCENARIO_ID=01 NUM_CLIENTS=3            # quick local smoke test (3 clients)
make native-run SCENARIO_ID=01 CLIENT_DEVICE=mps        # use Apple Silicon GPU
```

Scenario IDs correspond to entries in `matrix.yaml`. The Makefile merges `config.yaml` + the scenario's overrides before launching.

### Training (Docker — single machine)

```bash
make docker-run NUM_CLIENTS=11 SCENARIO_ID=01   # server + 11 clients in containers
make docker-clean                                # teardown
```

### Training (Distributed — VPS server + 11 Raspberry Pis)

Prerequisites: Tailscale overlay network set up, `ansible/inventory.ini` populated, Docker image pushed to Docker Hub.

```bash
# First time or after code changes: build and push a new image
make dist-build IMAGE_TAG=sha-abc123

# Sync configs, restart server on VPS, deploy clients to all Pis
make dist-start IMAGE_TAG=sha-abc123

# Follow live server logs
make dist-logs

# Fetch results back to Mac when done
make dist-fetch
```

For a full experiment matrix over the cluster:

```bash
make matrix BACKEND=dist
```

Each scenario's merged config is automatically pushed to VPS and all Pis before that scenario starts — clients always run with the correct overrides applied.

To wipe everything and start fresh:

```bash
make dist-restart
```

### Evaluation & Plotting

```bash
make eval-latest                              # evaluate latest session
make eval-session SESSION=<session-id>        # evaluate specific session
make plot-latest                              # plot latest session
make plot-session SESSION=<session-id>        # plot specific session
```

### Experiment Matrix

Scenarios and seeds are defined in `matrix.yaml`. Each scenario overrides `config.yaml` for that run only — processes never see the original base config during a matrix run.

```bash
make matrix-dry-run                        # preview scenario plan
make matrix                                # run all 14 scenarios × 3 seeds (native)
make matrix BACKEND=dist                   # run on VPS + Pi cluster
make matrix ONLY=01,09 MAX_RUNS=2          # run selected scenarios only
make matrix MATRIX_CONFIG=my_matrix.yaml   # use a custom matrix file
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

## Configuration

Two config files, two responsibilities:

**`config.yaml`** — runtime config; what a single direct run actually uses:

| Section         | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `model.*`       | LSTM hidden size, layers, sequence length, horizon                  |
| `training.*`    | Learning rate, rounds, local steps, loss weights, focal loss params |
| `compression.*` | Default mode, top-k ratio                                           |
| `federated.*`   | Number of clients, base ρ                                           |
| `scheduler.*`   | Latency thresholds, ρ bounds, EMA alpha                             |
| `profiler.*`    | Synthetic latency generator (base, offsets, jitter, burst)          |
| `data.*`        | Dataset paths, train/val/test split boundaries                      |

**`matrix.yaml`** — experiment matrix; only read by `make matrix`:

| Section                        | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| `experiment_matrix.seeds`      | Random seeds for repeated runs                     |
| `experiment_matrix.runner.*`   | Backend (native/docker), devices, timeout          |
| `experiment_matrix.scenarios`  | 14 scenario definitions with per-scenario overrides |

Each scenario in `matrix.yaml` deep-merges its `overrides` onto `config.yaml` to produce a fully resolved per-run config. Training processes only ever see this merged result — they never read `matrix.yaml` directly.

---

## Experiment Matrix (01–14)

Defined in `matrix.yaml`. 14 scenarios total (no-latency: S0-S3; mid/high: S0-S4), run with 3 seeds each.

| Latency Profile  | ID  | Strategy (S)                        | Compression | ρ       | Scheduler |
| ---------------- | --- | ----------------------------------- | ----------- | ------- | --------- |
| No latency       | 01  | S0 — baseline                       | float32     | 1       | Off       |
| No latency       | 02  | S1 — compression only (tier 1)      | float16     | 1       | Off       |
| No latency       | 03  | S2 — compression only (tier 2)      | int8        | 1       | Off       |
| No latency       | 04  | S3 — sync interval only             | float32     | 3       | Off       |
| Mid (~8 ms)      | 05  | S0 — baseline                       | float32     | 1       | Off       |
| Mid (~8 ms)      | 06  | S1 — compression only (tier 1)      | float16     | 1       | Off       |
| Mid (~8 ms)      | 07  | S2 — compression only (tier 2)      | int8        | 1       | Off       |
| Mid (~8 ms)      | 08  | S3 — sync interval only             | float32     | 3       | Off       |
| Mid (~8 ms)      | 09  | S4 — adaptive (joint)               | float32     | dynamic | On        |
| High (~50 ms)    | 10  | S0 — baseline                       | float32     | 1       | Off       |
| High (~50 ms)    | 11  | S1 — compression only (tier 1)      | float16     | 1       | Off       |
| High (~50 ms)    | 12  | S2 — compression only (tier 2)      | int8        | 1       | Off       |
| High (~50 ms)    | 13  | S3 — sync interval only             | float32     | 3       | Off       |
| High (~50 ms)    | 14  | S4 — adaptive (joint)               | float32     | dynamic | On        |

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
