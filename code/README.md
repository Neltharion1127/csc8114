# Federated Split Learning for Rainfall Prediction

A split-learning + federated-aggregation system for 24-hour rainfall prediction across 11 weather stations (Newcastle area, UK). The neural network is split between edge clients (LSTM encoder) and a central server (MLP predictor). Clients transmit compressed intermediate activations over gRPC rather than raw data. An adaptive scheduler jointly controls compression mode (float32 / float16 / int8) and sync interval (ρ) based on measured per-step network latency.

## Quick Start

```bash
uv sync                  # install dependencies
make compile-proto       # regenerate gRPC stubs (only needed if proto changes)
make download-data       # download and preprocess sensor data
make native-run          # train: 1 server + 11 clients locally
```

## Dataset

- **Source**: Open-Meteo historical weather API, 11 stations around Newcastle upon Tyne
- **Period**: 2015-01-01 – 2026-03-31 (hourly)
- **Target**: sum of rainfall over the next 24 h (`future_rain`)
- **Split** (absolute date boundaries, configured in `config.yaml`):

| Phase | Range | Rows/sensor |
|-------|-------|-------------|
| TRAIN | < 2024-01-01 | ~78 900 |
| VAL   | 2024-01-01 – 2024-12-31 | ~8 800 |
| TEST  | ≥ 2025-01-01 | ~10 900 |

## Training

### Native (recommended for development)

```bash
make native-clean
make native-run                                         # uses config.yaml defaults
make native-run NUM_CLIENTS=3 CLIENT_DEVICE=mps         # override device
```

### Docker (single-machine simulation)

```bash
make docker-run NUM_CLIENTS=11
```

### Distributed (VPS server + Raspberry Pi clients)

```bash
make dist-start
```

## Experiment Matrix

17 scenarios defined in `matrix.yaml`, covering three latency conditions × compression mode × sync interval (ρ):

| Prefix | Latency | Scenarios |
|--------|---------|-----------|
| N01–N04 | No latency (profiler off) | baseline, float16, int8, ρ=3 |
| L05–L10 | Low (~8 ms) | float32, float16, int8, ρ=3, Adaptive, Async |
| H11–H16 | High (~50 ms) | float32, float16, int8, ρ=3, Adaptive, Async |
| M17 | Mixed (per-client offsets) | Adaptive |

```bash
make matrix                          # run all 17 scenarios (3 seeds each)
make matrix ONLY=L09,H15 MAX_RUNS=1  # run selected scenarios
make matrix-dry-run                  # preview without running
```

Results land in `results/<session>/<scenario>/` and `bestweights/<session>/<scenario>/`.

## Evaluation

```bash
# Evaluate one scenario (threshold fixed at 0.38)
uv run python src/data/run_evaluation.py \
    --session 2026-04-30_01-17-30 \
    --scenario N01 \
    --force-prob-threshold 0.38 \
    --report-tag fixed38 \
    --device mps

# Re-evaluate all 17 scenarios in one go
bash run_eval_all.sh

# Rebuild matrix_summary.csv from all eval reports
uv run python src/data/build_matrix_summary.py
```

### Current sessions

| Session | Scenarios |
|---------|-----------|
| `2026-04-30_01-17-30` | N01–N04, L05–L09, H11–H14 |
| `2026-04-30_16-04-59` | L10, H15, H16, M17 |

Eval reports used for analysis: `*_eval_report_fixed38.csv` (threshold = 0.38, derived from N01 val scan).

### Key metrics

- **AUPRC** — primary classification metric (threshold-free)
- **ROC-AUC** — secondary classification metric
- **F1** — at threshold 0.38
- **MSE / MAE** — over all test samples (including dry days); `rain_mse` / `rain_mae` columns report rain-event-only errors

## Paper Figures

All scripts are in `src/data/`. Run from `code/`:

```bash
uv run python src/data/plot_compression_auprc.py      # Fig 2: AUPRC by compression & latency
uv run python src/data/plot_efficiency_accuracy.py    # Fig 3: accuracy–bandwidth scatter
uv run python src/data/plot_rho_convergence.py        # Fig 5: rho=1 vs rho=3 convergence
uv run python src/data/plot_scheduler_timeline.py     # Fig A: adaptive scheduler timeline
uv run python src/data/plot_monthly_performance.py    # Fig B: monthly F1 & MSE (Apr 2025–Mar 2026)
```

Outputs go to `results/graphics/` as both `.pdf` (for LaTeX) and `.png` (preview).

## Output Structure

```
bestweights/<session>/<scenario>/
  best_client_<i>_round_<r>_model_<ts>.pth   # best checkpoint per client
  periodic/
    client_<i>_round_<r>.pth                 # per-round paired checkpoints
    server_round_<r>.pth

results/<session>/
  <scenario>/
    training_log_client<i>_<ts>.csv          # step-level: loss, latency, payload
    training_log_client<i>_meta.json         # best checkpoint path + config
    server_log_<session>.csv                 # per-round aggregation
  <scenario>_eval_report_fixed38.csv/.json   # test metrics (source of truth)
  graphics/
    fig2_compression_auprc.pdf/.png
    fig3_efficiency_accuracy.pdf/.png
    fig5_rho_convergence.pdf/.png
    figA_scheduler_timeline.pdf/.png
    figB_monthly_performance.pdf/.png
  matrix_summary.csv                         # one row per scenario, all key metrics
```

## Key Source Files

| File | Role |
|------|------|
| `src/nodes/server_node.py` | gRPC server entry point |
| `src/nodes/client_node.py` | gRPC client entry point |
| `src/server/forward_service.py` | Forward RPC: loss, gradient, scheduler directives |
| `src/server/fedavg.py` | FedAvg coordinator with barrier + grace period |
| `src/server/scheduler.py` | Adaptive compression/ρ state machine |
| `src/client/training_loop.py` | Train/val epochs, checkpoint tracking |
| `src/client/forward_step.py` | Forward/backward RPC calls |
| `src/shared/compression.py` | float32 / float16 / int8 compression modes |
| `src/data/run_experiment_matrix.py` | Spawns 17-scenario ablation matrix |
| `src/data/run_evaluation.py` | Offline test evaluation |
| `src/data/build_matrix_summary.py` | Aggregates eval reports → matrix_summary.csv |

## Configuration (`config.yaml`)

Key sections:

```yaml
data:
  train_end: "2024-01-01"   # exclusive upper bound for TRAIN phase
  val_end:   "2025-01-01"   # exclusive upper bound for VAL phase

model:
  hidden_size: 64
  num_layers: 2
  seq_len: 48               # 48-hour input window
  input_size: 5

training:
  lr: 0.0005
  num_rounds: 30
  target_transform: log1p   # rainfall target in log1p space
  rain_threshold_mm: 0.1
  rain_probability_threshold: 0.38

federated:
  num_clients: 11
  min_clients_per_round: 9
  rho: 1                    # sync interval (steps between FedAvg rounds)

compression:
  mode: float32             # float32 | float16 | int8

scheduler:
  enabled: true
  float16_threshold_ms: 4.0
  int8_threshold_ms: 10.0
  ema_alpha: 0.2
```
