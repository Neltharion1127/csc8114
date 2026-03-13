# Federated Split Learning for Rainfall Prediction

This project implements a split-learning + federated-aggregation pipeline for rainfall prediction using multi-sensor weather time series.

Current default workflow is Makefile-driven and supports:
- Native run: one local server process + N local client processes
- Docker run: one server container + N client containers
- Auto plotting after training

## 1. Environment Setup

```bash
uv sync
```

## 2. Key Commands

Run `make help` for the full list.

### Proto / Data

```bash
make compile-proto
make download-data
```

- `compile-proto`: regenerate gRPC Python stubs from `proto/fsl.proto`
- `download-data`: download and preprocess data via `src.data.data_download_openmeteo`

### Training (Recommended: Native)

```bash
make clean-native
make run-native
```

Notes:
- `run-native` starts server first, waits for readiness, then starts all clients.
- Log output is color-prefixed by process (`server`, `client-1`, `client-2`, ...).
- Training completion triggers automatic plotting (`make plot-latest`).

Useful overrides:

```bash
make run-native SERVER_HOST=127.0.0.1 SERVER_PORT=50051 SERVER_DEVICE=cpu CLIENT_DEVICE=mps
```

### Training (Docker)

```bash
make run-network
```

Alias:

```bash
make test-network
```

Cleanup:

```bash
make clean
```

## 3. Plotting

Plot latest session:

```bash
make plot-latest
```

Plot one specific session:

```bash
make plot-session SESSION=2026-03-13_03-19-17
```

Only confusion matrix:

```bash
make plot-confusion SESSION=2026-03-13_03-19-17
```

Generated outputs are written under `results/<session>/`.

## 4. Runtime Structure

- Server entry: `src/nodes/server_node.py`
- Client entry: `src/nodes/client_node.py`

Core modules:
- Client flow: `src/client/*` (`training_loop.py`, `forward_step.py`, `sync.py`, `checkpointing.py`, `reporting.py`)
- Server flow: `src/server/*` (`forward_service.py`, `fedavg.py`, `scheduler.py`, `reporting.py`, `bootstrap.py`)

## 5. Important Configs (`config.yaml`)

### Training
- `training.num_rounds`
- `training.local_steps`
- `training.lr`
- `training.seed`
- `training.early_stopping_patience`
- `training.eval_max_samples_per_sensor`

### Classification Behavior
- `training.classification_loss_type`: `weighted_bce` or `focal`
- `training.focal_gamma`
- `training.focal_alpha`
- `training.classification_positive_weight`
- `training.rain_threshold_mm`
- `training.rain_probability_threshold`

### System
- `federated.num_clients`
- `scheduler.enabled`
- `profiler.enabled`
- `server.log_flush_interval`

## 6. Output Directories

- Model checkpoints: `bestweights/<session>/`
- Client logs and metadata: `results/<session>/training_log_client*.csv` and `*_meta.json`
- Server logs: `results/<session>/server_log_<session>.csv`
- Plots:
  - `training_curve_<session>.png`
  - `server_metrics_<session>.png`
  - `confusion_matrix_<session>.png`
  - `confusion_matrix_metrics_<session>.csv`
