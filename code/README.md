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
- `experiment_matrix.*` (batch experiment definitions and runner settings)
- `FSL_CONFIG_PATH` (optional env var to run with an alternate YAML config)
- `experiment_matrix.runner.backend`: `native` or `docker`

## 6. Output Directories

- Model checkpoints: `bestweights/<session>/`
- Client logs and metadata: `results/<session>/training_log_client*.csv` and `*_meta.json`
- Server logs: `results/<session>/server_log_<session>.csv`
- Plots:
  - `training_curve_<session>.png`
  - `server_metrics_<session>.png`
  - `confusion_matrix_<session>.png`
  - `confusion_matrix_metrics_<session>.csv`

## 7. Latest Baseline (Session `2026-03-13_21-45-05`)

This is the current reference run after tightening train/val/test logic and strict checkpoint pairing.

Time windows used in this run:
- Train: `< 2026-02-10 00:00:00`
- Validation: `[2026-02-10 00:00:00, 2026-02-24 00:00:00)`
- Test: `>= 2026-02-24 00:00:00`

Final test metrics (`results/2026-03-13_21-45-05/evaluation_report_2026-03-13_21-45-05.csv`):
- Client 1: Recall `0.4495`, Precision `0.6106`, F1 `0.5178`
- Client 2: Recall `0.4691`, Precision `0.6486`, F1 `0.5444`
- Client 3: Recall `0.4710`, Precision `0.6008`, F1 `0.5280`
- Average: Recall `0.4632`, Precision `0.6200`, F1 `0.5301`

Config highlights used in this baseline (`config.yaml`):
- `data.test_days: 14`
- `data.val_days: 14`
- `training.classification_loss_type: focal`
- `training.classification_loss_weight: 2.0`
- `training.classification_positive_weight: 1.1`
- `training.rain_probability_threshold: 0.34`
- `training.rain_sample_ratio: 0.15`
- `training.regression_loss_weight: 1.0`

## 8. How To Use Result Data

Recommended order for analysis/reporting:
- Use `evaluation_report_<session>.json` / `.csv` as the source of truth for final TEST metrics.
- Use `training_log_client*.csv` for training/validation behavior and sample-level debugging.
- Use `server_log_<session>.csv` for aggregation/system diagnostics.
- Use `training_log_client*_meta.json` to recover exact run config and best checkpoint path.

Important files for session `2026-03-13_21-45-05`:
- `results/2026-03-13_21-45-05/evaluation_report_2026-03-13_21-45-05.json`
- `results/2026-03-13_21-45-05/evaluation_report_2026-03-13_21-45-05.csv`
- `results/2026-03-13_21-45-05/training_log_client1_20260313_214619.csv` (and client2/client3 equivalents)
- `results/2026-03-13_21-45-05/server_log_2026-03-13_21-45-05.csv`
- `bestweights/2026-03-13_21-45-05/periodic/` (strictly paired per-round checkpoints)

Strict evaluation behavior:
- `run_evaluation.py` first tries strict periodic pairing (`server_round_R` + all `client_i_round_R`).
- If strict pairing exists, it evaluates that exact round as `pairing_mode=periodic_round_R`.
- Classification metrics are recomputed offline by default (`cls_metric_source=offline_recomputed`) unless checkpoint metrics are explicitly marked as TEST and sample counts match.
- Evaluation loads thresholds and related settings from the checkpoint `config_snapshot` when available, to avoid config drift.

Useful commands:

Evaluate latest bestweights session:
```bash
make eval-latest
```

Evaluate a specific session:
```bash
make eval-session SESSION=2026-03-13_21-45-05 PLOT_DEVICE=cpu
```

Evaluate a specific round with strict pairing:
```bash
python -m src.data.run_evaluation --session 2026-03-13_21-45-05 --round 30 --device cpu
```

Generate plots for a specific session:
```bash
make plot-session SESSION=2026-03-13_21-45-05 PLOT_DEVICE=cpu
```

Dry-run matrix plan from `config.yaml`:
```bash
make matrix-dry-run
```

Run full matrix:
```bash
make matrix
```

Run selected scenarios only:
```bash
make matrix ONLY=M01,M10 MAX_RUNS=2
```

Run matrix on Docker backend:
```bash
make matrix BACKEND=docker
```
