# CSC8114 Group 9 — Joint Computation and Communication Optimisation in Federated Split Learning for IoT Rainfall Prediction

> A research prototype / experimental platform Federated Split Learning system with adaptive communication control for real-time IoT rainfall prediction.

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

- **Reproducible Experiment Matrix** — 17 carefully designed scenarios × 3 random seeds, systematically decoupling the effects of latency profiles (none / low / high / mixed), compression modes, synchronisation intervals, and adaptive vs. static policies. Scenarios are defined in `matrix.yaml` and run with a single `make matrix` command; `config.yaml` remains the clean single-run baseline.

### Key Results

Measured across 17 scenarios × 3 seeds; baseline is N01 (float32, no latency, ρ=1, F1=0.520, AUPRC=0.649).

| Finding | Detail |
| ------- | ------ |
| INT8 compression (no latency) | **−73.4% payload** (256 B → 68 B), AUPRC 0.649 → 0.651 (stable) |
| ρ = 3 sync interval (no latency) | AUPRC 0.649 → 0.661 (+0.012); fewer aggregation rounds, similar quality |
| Adaptive under low latency (~8 ms) | **−63.9% payload** (256 B → 92 B), AUPRC 0.649 → 0.648 (stable) |
| Adaptive under high latency (~50 ms) | **−73.4% payload** (256 B → 68 B), AUPRC 0.649 → 0.650 (stable) |
| AUPRC stability | Ranges 0.648–0.661 across all 17 scenarios — quality robust to compression and latency |

Compression reduces bandwidth by up to 73% with negligible AUPRC loss (<0.002). The adaptive scheduler automatically selects the appropriate compression tier (float16 under low latency, int8 under high latency) without manual tuning.

---

## Repository Structure

```
csc8114/
├── code/                        # FSL system implementation
│   ├── config.yaml              # Runtime config — model, training, data, comms
│   ├── matrix.yaml              # Experiment matrix — 17 scenarios & seeds (ablation only)
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
make native-run                                         # uses config.yaml defaults
make native-run NUM_CLIENTS=3                           # quick local smoke test
make native-run CLIENT_DEVICE=mps                       # use Apple Silicon GPU
```

### Training (Docker — single machine)

```bash
make docker-run NUM_CLIENTS=11   # server + 11 clients in containers
make docker-clean                # teardown
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
# Re-evaluate all 17 scenarios
bash run_eval_all.sh

# Or evaluate a single scenario
uv run python src/data/run_evaluation.py \
    --session 2026-05-03_00-20-00 --scenario N01 \
    --eval-max-samples 0 --device mps

# Aggregate all eval reports into one CSV
uv run python src/data/build_matrix_summary.py

# Regenerate paper figures (output → results/graphics/)
uv run python src/data/plot_compression_auprc.py      # Fig 2
uv run python src/data/plot_efficiency_accuracy.py    # Fig 3
uv run python src/data/plot_rho_convergence.py        # Fig 5
uv run python src/data/plot_scheduler_timeline.py     # Fig A
uv run python src/data/plot_monthly_performance.py    # Fig B
```

### Experiment Matrix

Scenarios and seeds are defined in `matrix.yaml`. Each scenario overrides `config.yaml` for that run only — processes never see the original base config during a matrix run.

```bash
make matrix-dry-run                        # preview scenario plan
make matrix                                # run all 17 scenarios × 3 seeds (native)
make matrix ONLY=L09,H15 MAX_RUNS=1        # run selected scenarios only
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
│              gRPC Clients (×11)                         │
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
| `experiment_matrix.scenarios`  | 17 scenario definitions with per-scenario overrides |

Each scenario in `matrix.yaml` deep-merges its `overrides` onto `config.yaml` to produce a fully resolved per-run config. Training processes only ever see this merged result — they never read `matrix.yaml` directly.

---

## Experiment Matrix (N01–M17)

Defined in `matrix.yaml`. 17 scenarios total, run with 3 seeds each. Results land in `results/<session>/<scenario>/` and `bestweights/<session>/<scenario>/`.

| ID  | Latency Profile   | Compression | ρ       | Scheduler      | AUPRC  | F1     | Payload |
| --- | ----------------- | ----------- | ------- | -------------- | ------ | ------ | ------- |
| N01 | No latency        | float32     | 1       | Off            | 0.6490 | 0.5199 | 256 B   |
| N02 | No latency        | float16     | 1       | Off            | 0.6510 | 0.5266 | 128 B   |
| N03 | No latency        | int8        | 1       | Off            | 0.6513 | 0.6089 | 68 B    |
| N04 | No latency        | float32     | 3       | Off            | 0.6608 | 0.5729 | 256 B   |
| L05 | Low (~8 ms)       | float32     | 1       | Off            | 0.6493 | 0.5267 | 256 B   |
| L06 | Low (~8 ms)       | float16     | 1       | Off            | 0.6496 | 0.5187 | 128 B   |
| L07 | Low (~8 ms)       | int8        | 1       | Off            | 0.6490 | 0.5331 | 68 B    |
| L08 | Low (~8 ms)       | float32     | 3       | Off            | 0.6592 | 0.5544 | 256 B   |
| L09 | Low (~8 ms)       | dynamic     | 1       | Adaptive       | 0.6482 | 0.5200 | 92 B    |
| L10 | Low (~8 ms)       | dynamic     | dynamic | Adaptive + ρ   | 0.6577 | 0.5747 | 92 B    |
| H11 | High (~50 ms)     | float32     | 1       | Off            | 0.6490 | 0.5580 | 256 B   |
| H12 | High (~50 ms)     | float16     | 1       | Off            | 0.6483 | 0.5246 | 128 B   |
| H13 | High (~50 ms)     | int8        | 1       | Off            | 0.6486 | 0.5078 | 68 B    |
| H14 | High (~50 ms)     | float32     | 3       | Off            | 0.6554 | 0.5426 | 256 B   |
| H15 | High (~50 ms)     | dynamic     | 1       | Adaptive       | 0.6497 | 0.6277 | 68 B    |
| H16 | High (~50 ms)     | dynamic     | dynamic | Adaptive + ρ   | 0.6567 | 0.5525 | 68 B    |
| M17 | Mixed (per-client)| dynamic     | dynamic | Adaptive + ρ   | 0.6571 | 0.6066 | 127 B   |

---

## Output Structure

Each training session produces:

```
bestweights/<session>/<scenario>/
  ├── best_client_<i>_round_<r>_model_<ts>.pth   # best checkpoint per client
  └── periodic/
      ├── client_<i>_round_<r>.pth               # per-round paired checkpoints
      └── server_round_<r>.pth

results/<session>/
  ├── <scenario>/
  │   ├── training_log_client<i>_<ts>.csv         # step-level: loss, latency, payload
  │   ├── training_log_client<i>_meta.json        # best checkpoint path + config
  │   └── server_log_<session>.csv                # per-round aggregation
  ├── <scenario>_eval_report.csv/.json             # test metrics (source of truth)
  ├── graphics/                                   # paper figures (pdf + png)
  └── matrix_summary.csv                          # one row per scenario, all key metrics
```

---

## Paper

The IEEE conference paper is in `paper/`. To compile:

```bash
cd paper/
pdflatex csc8114.tex
bibtex csc8114
pdflatex csc8114.tex
pdflatex csc8114.tex
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
