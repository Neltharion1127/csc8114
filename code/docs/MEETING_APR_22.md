# Meeting — April 22, 2026

## Background

This project implements a **Federated Split Learning (FSL)** system for IoT-based rainfall prediction over Newcastle weather data (2015–2026). Neural networks are split between 11 edge clients (a 2-layer LSTM encoder) and a central server (an MLP predictor with dual regression/classification heads). Clients transmit compressed intermediate activations (smashed data) instead of raw sensor readings, preserving data privacy while reducing communication overhead.

The central contribution is an **adaptive scheduler** that jointly controls:
- **Spatial compression**: float32 → float16 → int8 → topk\_int8 (TopK sparsification + int8 quantization), triggered by EMA-smoothed per-client latency thresholds (4 ms / 10 ms / 15 ms). The 15 ms threshold is derived as `int8_threshold × topk_multiplier = 10 × 1.5`.
- **Temporal sync interval ρ**: how many local forward steps occur between FedAvg aggregation rounds

The system targets three deployment conditions calibrated against real IoT literature (Mao et al. 2017): none (~0 ms, LAN/WiFi baseline), low latency (~8–14 ms, 4G MEC), and high latency (~50–80 ms, NB-IoT / wide-area IoT).

**Prediction task:** A sliding window of the past 48 hours (seq_len=48) is used as input. The target is a binary label derived from the cumulative rainfall over the **next 24 hours** (horizon=24): labelled 1 (rain) if the 24 h sum ≥ 0.5 mm (`rain_threshold_mm`), otherwise 0 (no rain). The model outputs a rain probability via sigmoid; predictions are classified as rain if the probability exceeds 0.35 (`rain_probability_threshold`). The server head has two outputs — a classification head (used for F1/AUPRC) and a regression head that predicts rainfall amount in mm (log1p-transformed during training, inverse-transformed for MSE/MAE evaluation).

Dataset: Open Meteo, 11 Newcastle stations (Gateshead excluded)
Hourly resolution, 98,592 rows per station (2015-01-01 → 2026-03-31)

| Split | Range | Rows/station | Share |
|---|---|---|---|
| TRAIN | 2015-01-01 → 2024-12-31 | 87,672 | 88.9% |
| VAL   | 2025-01-01 → 2025-06-30 | 4,344  | 4.4%  |
| TEST  | 2025-07-01 → 2026-03-31 | 6,576  | 6.7%  |

---

## Progress

Since the last meeting we have finalised the experiment matrix under two deployment conditions: **native simulation** (single-machine, synthetic latency injected via a profiler) and **Raspberry Pi** (real hardware clients over Tailscale VPN). The matrix covers three design axes:

| Axis | Levels |
|---|---|
| Compression strategy | float32 (baseline) · float16 · int8 · adaptive (scheduler-controlled) |
| Sync interval ρ | 1 (sync every round) · 3 (less frequent) · dynamic (scheduler-controlled) |
| Latency regime | None (~0 ms) · Low (~8–14 ms) · High (~50–80 ms) |

### Table 1 — Simulation Experiment Matrix (14 scenarios × 3 seeds = 42 runs, all completed)

| ID | Latency Regime | Strategy | Compression | ρ | Scheduler | Seeds |
|---|---|---|---|---|---|---|
| M01 | None | S0 — Baseline | float32 | 1 | Off | 42, 52, 62 |
| M02 | None | S1 — Compression only | float16 | 1 | Off | 42, 52, 62 |
| M03 | None | S2 — Compression only | int8 | 1 | Off | 42, 52, 62 |
| M04 | None | S3 — Sync interval only | float32 | 3 | Off | 42, 52, 62 |
| M05 | Low (~8–14 ms) | S0 — Baseline | float32 | 1 | Off | 42, 52, 62 |
| M06 | Low (~8–14 ms) | S1 — Compression only | float16 | 1 | Off | 42, 52, 62 |
| M07 | Low (~8–14 ms) | S2 — Compression only | int8 | 1 | Off | 42, 52, 62 |
| M08 | Low (~8–14 ms) | S3 — Sync interval only | float32 | 3 | Off | 42, 52, 62 |
| M09 | Low (~8–14 ms) | S4 — Adaptive (compression only, ρ fixed) | auto | 1 | On | 42, 52, 62 |
| M10 | High (~50–80 ms) | S0 — Baseline | float32 | 1 | Off | 42, 52, 62 |
| M11 | High (~50–80 ms) | S1 — Compression only | float16 | 1 | Off | 42, 52, 62 |
| M12 | High (~50–80 ms) | S2 — Compression only | int8 | 1 | Off | 42, 52, 62 |
| M13 | High (~50–80 ms) | S3 — Sync interval only | float32 | 3 | Off | 42, 52, 62 |
| M14 | High (~50–80 ms) | S4 — Adaptive (compression only, ρ fixed) | auto | 1 | On | 42, 52, 62 |

### Table 2 — Raspberry Pi Experiment Matrix (10 of 14 scenarios × 3 seeds = 30 runs completed)

| ID | Latency Regime | Strategy | Status |
|---|---|---|---|
| M01 | None | S0 Baseline | Done |
| M02 | None | S1 float16 | Done |
| M03 | None | S2 int8 | **Pending** |
| M04 | None | S3 rho=3 | Done |
| M05 | Low | S0 Baseline | Done |
| M06 | Low | S1 float16 | Done |
| M07 | Low | S2 int8 | **Pending** |
| M08 | Low | S3 rho=3 | Done |
| M09 | Low | S4 Adaptive | Done |
| M10 | High | S0 Baseline | Done |
| M11 | High | S1 float16 | Done |
| M12 | High | S2 int8 | **Pending** |
| M13 | High | S3 rho=3 | Done |
| M14 | High | S4 Adaptive | **Pending** |

---

## Results

### Simulation Results (all 14 scenarios, averaged across 3 seeds)

Macro F1 is sensitive to model collapse (all-positive prediction); AUPRC is reported as the primary reliability metric since the dataset is class-imbalanced (~53% rain). Payload is the per-step compressed activation size transmitted per client.

Payload for M01–M04 is theoretical (profiler disabled, not measured); M05–M14 are measured averages.

| ID | Latency (ms) | Payload (B) | Compression Modes Seen | Macro F1 | AUPRC |
|---|---|---|---|---|---|
| M01 | 0 | 256† | float32 | 0.596 | 0.701 |
| M02 | 0 | 128† | float16 | 0.651 | 0.700 |
| M03 | 0 | 68†  | int8 | 0.646 | 0.702 |
| M04 | 0 | 256† | float32 | 0.244 | 0.689 |
| M05 | 10.6 | 256 | float32 | 0.673 | 0.710 |
| M06 | 10.7 | 128 | float16 | 0.636 | 0.709 |
| M07 | 10.7 | 68 | int8 | 0.538 | 0.702 |
| M08 | 12.4 | 256 | float32 | 0.147 | 0.682 |
| **M09** | **10.7** | **93** | float32/float16/int8/topk_int8 | **0.699** | **0.701** |
| M10 | 63.0 | 256 | float32 | 0.344 | 0.704 |
| M11 | 63.2 | 128 | float16 | 0.553 | 0.699 |
| M12 | 62.9 | 68 | int8 | 0.603 | 0.699 |
| M13 | 63.0 | 256 | float32 | 0.030 | 0.678 |
| **M14** | **63.3** | **52** | float32/topk_int8 | **0.672** | **0.711** |

**Key observations:**

- **Sync interval alone (S3) is harmful.** M04/M08/M13 all collapse in Macro F1. Under high latency (M13), F1 drops to 0.030 — the model barely converges before FedAvg barrier timeouts accumulate.
- **Adaptive scheduler recovers accuracy under high latency.** M14 (adaptive) achieves F1=0.672 vs M10 (float32 baseline, same latency) F1=0.344 — a 96% relative improvement — while payload drops from 256 B to 52 B (80% bandwidth reduction).
- **Low-latency adaptive (M09) achieves the best overall balance.** F1=0.699 with 63% lower payload than the M05 baseline; the scheduler dynamically selects float32/float16/int8/topk_int8 per client based on measured EMA latency.
- **AUPRC is stable across compression modes** (~0.678–0.711), confirming that probabilistic ranking quality is preserved even under aggressive quantisation. The main risk is binary-decision F1 collapse, not ranking degradation.

### Raspberry Pi Results (partial — seed 42, 11-client averages)

Results are per-client averages from real Raspberry Pi hardware. M03, M07, M12, M14 are pending.

| Scenario (Pi) | Strategy | Avg Accuracy | Avg F1 | Avg AUPRC | Notes |
|---|---|---|---|---|---|
| M01 | S0 float32, no lat | 53.6% | 0.697 | 0.690 | Model collapse — all-positive predictions |
| M02 | S1 float16, no lat | — | — | — | Available, aggregation in progress |
| M04 | S3 rho=3, no lat | — | — | — | Available, aggregation in progress |
| M05 | S0 float32, low lat | — | — | — | Available, aggregation in progress |
| M06 | S1 float16, low lat | — | — | — | Available, aggregation in progress |
| M08 | S3 rho=3, low lat | — | — | — | Available, aggregation in progress |
| **M09** | **S4 adaptive, low lat** | **62.9%** | **0.619** | **0.708** | **No collapse; real discrimination on real hardware** |
| M10 | S0 float32, high lat | — | — | — | Available, aggregation in progress |
| M11 | S1 float16, high lat | — | — | — | Available, aggregation in progress |
| M13 | S3 rho=3, high lat | — | — | — | Available, aggregation in progress |

**Pi-specific observations:**

- **Pi M09 (adaptive) avoids model collapse** and achieves AUPRC=0.708, consistent with simulation (0.701). This confirms the adaptive scheduler operates correctly on real edge hardware with real network jitter.
- **Pi M01 baseline degenerates** to all-positive prediction (recall=1.0, precision=53.6%), matching simulation behaviour (2 of 3 seeds also degenerated). This is expected under class imbalance without adaptive compression.
- Full seed 52/62 aggregation and remaining scenario parsing are in progress.

---

## Next Steps

Clean the results and prepare for writing