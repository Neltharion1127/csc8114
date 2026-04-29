# Meeting — April 22, 2026 Updates

### 1. Enhanced System Observability & Efficiency Metrics
In this latest update, we add our evaluation matrix: while maintaining the requirement that the model learns effectively, we also focus on quantifying the **resource footprint** of the Federated Split Learning process. This allows us to empirically evaluate the trade-offs between predictive accuracy and system-wide efficiency.
*   **Hardware Telemetry**: Integrated `psutil` into the client training loop to log **CPU Usage (%)** and **Memory Usage (%)** per epoch. 
*   **Communication & Model Size**: Added logic to calculate the exact **Model Size (Bytes)** and **Payload size** transmitted per forward step, allowing for a quantitative comparison of `float32` vs `int8` compression efficiency.
*   **System Throughput**: Added `Throughput (samples/s)` tracking in the evaluation reports. This allows us to prove that while the adaptive scheduler maintains accuracy, it significantly improves the **data processing density** by reducing communication-induced idle time.

### 2. Experimental matrix tuning

*   **Latency Calibration & Baseline Shift**: 
    *   Due to the observed higher-than-expected baseline latency on real hardware (~22ms), we have shifted the fixed-synchronization comparison group (PI03) from **rho=3 to rho=4**. This provides a more challenging and relevant baseline to prove if the adaptive scheduler can outperform a static, conservative sync policy.
    *   Re-tuned the `latency_threshold` to **20.0ms** and **40.0ms** to ensure the scheduler only escalates compression/rho when the network genuinely degrades beyond the physical baseline.
* Redesigned the experiment matrix to include more scenarios and for analysis on native and pi.
* Move native training to cloud

### 3. Modify Dataset Partitioning (2015–2026)
We have updated the chronological data split to follow a more standard 80/10/10 distribution over the extended timeframe:
*   **TRAIN (80.0%)**: 2015-01-01 to 2023-12-31.
*   **VAL (8.9%)**: 2024-01-01 to 2024-12-31 (Full year).
*   **TEST (11.1%)**: 2025-01-01 to 2026-03-31 (Final 15 months).


### 4. Research Hypothesis
We designed to systematically evaluate the trade-offs between **predictive performance** and **system efficiency** in latency-constrained IoT environments:
*   **Variable Impact**: By manipulating **quantization** (spatial compression) and **sync interval ρ** (temporal iteration), we observe their direct impact on training stability. For instance, we hypothesize that under high-latency conditions, aggressive compression may lead to a measurable decrease in **AUPRC**, but this is offset by significant improvements in communication speed.
*   **Resource Utility**: By incorporating the **resource footprint** (CPU, Mem, Throughput), we can quantitatively prove whether the adaptive scheduler successfully maximizes the "utility per unit of resource" on edge devices like the Raspberry Pi.
---

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
| TRAIN | 2015-01-01 → 2023-12-31 | 78,888 | 80.0% |
| VAL   | 2024-01-01 → 2024-12-31 | 8,784  | 8.9%  |
| TEST  | 2025-01-01 → 2026-03-31 | 10,920 | 11.1% |

---

Since the last meeting we have finalised the experiment matrix under two deployment conditions: **Native Simulation** (single-machine, synthetic latency injected via a profiler) and **Raspberry Pi** (real hardware clients over Tailscale VPN).

### Recent Optimizations (April 27 Updates)

To strengthen the empirical results for the thesis, we integrated deep hardware monitoring and refined the ablation parameters:

#### 1. Hardware & Performance Metrics
Integrated real-time system monitoring:
*   **CPU Usage (%)**: Monitor computational load on each Raspberry Pi.
*   **Memory Usage (%)**: Ensure memory footprint remains within edge device limits.
*   **Model Size (Bytes)**: Quantify the footprint of float32/float16/int8 weights to prove compression efficiency.
*   **Throughput & Timing**: Capture `Epoch_Time_s` and `Throughput (samples/s)` to demonstrate the latency-reduction benefits of the adaptive scheduler.


#### 2. Matrix Tuning for Simulation
Move to clould to training


#### 3. Matrix Tuning for Raspberry Pi
*   **Adaptive Thresholds**: Set `latency_threshold: 20.0ms`. Given the 22ms baseline RTT of the cluster, this ensures the adaptive scheduler is actively triggered.
*   **Fair Comparison Groups**: Scenario 04 (Sync-only) now has `rho` fixed at **4** to provide a direct head-to-head comparison with the highest intensity of the Adaptive scheduler (Scenario 09).


| Axis | Levels |
|---|---|
| Compression strategy | float32 (baseline) · float16 · int8 · S4 adaptive (compression) · S5 joint adaptive (compression + rho) |
| Sync interval ρ | 1 (sync every round) · 3 (fixed interval) · dynamic (scheduler-controlled severity) |
| Latency regime | None (~0 ms) · Low (~8 ms MEC) · High (~50 ms NB-IoT) |

### Table 1 — Simulation Experiment Matrix (16 scenarios × 3 seeds = 48 runs)

The simulation uses the internal latency profiler to emulate different network tiers.

| ID | Latency Regime | Strategy Variant | Compression | ρ | Scheduler |
|---|---|---|---|---|---|
| **N01** | None | S0 — Baseline | float32 | 1 | Off |
| **N02** | None | S1 — Comp. only | float16 | 1 | Off |
| **N03** | None | S2 — Comp. only | int8 | 1 | Off |
| **N04** | None | S3 — Sync only | float32 | 3 | Off |
| **L05** | Low (~8ms) | S0 — Baseline | float32 | 1 | Off |
| **L06** | Low (~8ms) | S1 — Comp. only | float16 | 1 | Off |
| **L07** | Low (~8ms) | S2 — Comp. only | int8 | 1 | Off |
| **L08** | Low (~8ms) | S3 — Sync only | float32 | 3 | Off |
| **L09** | Low (~8ms) | S4 — Adaptive (Comp) | auto | 1 | On (rho_step=0) |
| **L10** | Low (~8ms) | S5 — Joint Adaptive | auto | dynamic | On (rho_step=1) |
| **H11** | High (~50ms) | S0 — Baseline | float32 | 1 | Off |
| **H12** | High (~50ms) | S1 — Comp. only | float16 | 1 | Off |
| **H13** | High (~50ms) | S2 — Comp. only | int8 | 1 | Off |
| **H14** | High (~50ms) | S3 — Sync only | float32 | 3 | Off |
| **H15** | High (~50ms) | S4 — Adaptive (Comp) | auto | 1 | On (rho_step=0) |
| **H16** | High (~50ms) | S5 — Joint Adaptive | auto | dynamic | On (rho_step=1) |

### Table 2 — Raspberry Pi Experiment Matrix

Unlike simulation, Pi experiments do **not** use a latency profiler — real network latency between Raspberry Pi clients and the VPS server (over Tailscale) drives the system naturally. **Measured baseline RTT is ~22ms.** Two strategy variants are evaluated:

| ID | Strategy | Latency | Compression | ρ | Scheduler | Seeds | Status |
|---|---|---|---|---|---|---|---|
| PI01 | S0 — Baseline | Real (~22ms) | float32 | 1 | Off | 42,52,62 | **Running** |
| PI02 | S1 — Baseline | Real (~22ms) |float16 | 1 | Off | 42,52,62 | **Running** |
| PI03 | S2 — Baseline | Real (~22ms) | float32 | 4 | Off | 42,52,62 | **Running** |
| PI04 | S4 — Adaptive | Real (~22ms) | auto | 1→dynamic | On | 42,52,62 | **Running** |

---

## Metrics & Parameters

### Evaluation Metrics

| Metric | Meaning | Notes |
|---|---|---|
| **AUPRC** | Area Under the Precision-Recall Curve — ranking quality across all classification thresholds | Primary metric. Robust to class imbalance (~53% positive rate). Max = 1.0. |
| **Macro F1** | Harmonic mean of precision & recall, averaged equally across both classes | Values near 0.697 indicate model collapse (all-positive prediction — precision collapses to the positive class rate). |
| **MSE / MAE** | Regression head error in predicting rainfall amount (mm) | Measured after inverse log1p transform. Reflects quantity prediction quality, not just rain/no-rain. |
| **Accuracy** | Fraction of correct binary predictions | Misleading under imbalance — a model predicting all rain gets ~53% for free. |
| **Payload (B)** | Bytes transmitted per client per forward step (compressed activation) | Direct measure of communication cost. |
| **Latency (ms)** | Per-client round-trip time (RTT) | Measured via profiler in simulation; real network RTT on Raspberry Pi. |
| **Throughput (samples/s)** | Effective training throughput | Samples processed per second. |
| **Total Training Time (s)** | Total time to complete all federated rounds | End-to-end experiment duration. |
| **Model Size (MB)** | Client-side LSTM size in MB | Memory/storage footprint on edge devices. |
| **CPU/Mem Usage (%)** | Hardware resource utilization | Monitored real-time on Raspberry Pi to evaluate edge-efficiency. |


### System Parameters

| Parameter | Value | Meaning |
|---|---|---|
| **ρ (rho)** | 1 or 3 | Sync interval — local training rounds between FedAvg aggregations. ρ=1: sync every round; ρ=3: less frequent. Higher ρ reduces communication but delays weight sharing. |
| **Compression mode** | float32 / float16 / int8 / topk_int8 | How activations are encoded before transmission. float32=256 B; float16=128 B; int8=68 B; topk_int8≈52 B (top 12.5% elements, int8-quantised). |
| **EMA latency** | α=0.2 | Exponential moving average of per-client round-trip latency. Smooths noise before scheduler decisions. |
| **Scheduler thresholds** | 20 / 40 ms | EMA latency levels triggering adaptation: <20 ms→float32; 20–40 ms→Adaptive Rho; >40ms→int8. Calibrated against real Pi latency (~22ms). |
| **rho_step** | 1 | Per-severity ρ increment. ρ = base + severity. When latency=22ms, severity=1, thus ρ=2. |
| **rain_threshold_mm** | 0.5 mm | Minimum 24 h cumulative rainfall to label a sample as rain (target=1). |
| **prob_threshold** | 0.35 | Sigmoid output threshold for binary prediction. Set below 0.5 to improve recall on imbalanced data. |

---

## Results

### Simulation Results (16 scenarios, 3 seeds)

*Note: Results pending execution of the new 16-scenario matrix configuration.*
seeds = [42, 52, 62]

| ID | Strategy | F1 | AUC | Acc | MSE | MAE | Thrp | Model Size | Traff(payload) | CPU/M |
|---|---|---|---|---|---|---|---|---|---|---|
| **N01** | S0 Baseline | | | | | | | --MB | -- | |
| **N02** | S1 float16 | | | | | | | --MB | -- | |
| **N03** | S2 int8 | | | | | | | --MB | -- | |
| **N04** | S3 rho=3 | | | | | | | --MB | -- | |
| **L05** | S0 Baseline | | | | | | | --MB | -- | |
| **L06** | S1 float16 | | | | | | | --MB | -- | |
| **L07** | S2 int8 | | | | | | | --MB | -- | |
| **L08** | S3 rho=3 | | | | | | | --MB | -- | |
| **L09** | S4 Adaptive | | | | | | | --MB | -- | |
| **L10** | S5 Joint Adap | | | | | | | --MB | -- | |
| **H11** | S0 Baseline | | | | | | | --MB | -- | |
| **H12** | S1 float16 | | | | | | | --MB | -- | |
| **H13** | S2 int8 | | | | | | | --MB | -- | |
| **H14** | S3 rho=3 | | | | | | | --MB | -- | |
| **H15** | S4 Adaptive | | | | | | | --MB | -- | |
| **H16** | S5 Joint Adap | | | | | | | --MB | -- | |

**Key Observations (Ablation Analysis):**
*   **None (N01-N04)**: 
*   **Low (L05-L10)**: 
*   **High (H11-H16)**: 

### Raspberry Pi Results (Real-world cluster, seed 42, 11-client averages)

The hardware results below demonstrate the impact of real network jitter (Tailscale VPN) and resource constraints on the Raspberry Pi cluster.

| ID | Strategy Variant | F1 | AUC | Acc | MSE | MAE | Thrp | Size | Traff | CPU/M | 
|---|---|---|---|---|---|---|---|---|---|---|
| PI01 | S0 Baseline | | | | | | | --MB | -- | |
| PI02 | S1 float16 | | | | | | | --MB | -- | |
| PI03 | S2 Rho=4 | | | | | | | --MB | -- | |
| **PI04** | **S4 Adaptive** | | | | | | | --MB | -- | |

**Pi-specific Observations & Analysis:**

*   **Communication Efficiency**: (To be analysed) Does the adaptive scheduler achieve a higher throughput than the PI01 baseline in the 22ms environment?
*   **Accuracy Resilience**: (To be analysed) Does the S4 strategy avoid the "model collapse" observed in simulated high-latency baselines?
*   **Hardware Impact**: (To be analysed) Does the use of compression (float16/int8) and dynamic synchronization (rho) significantly affect the CPU/Memory utilization on the edge devices?


---

## Next Steps

1. **Experimental Run (Current Week)**: Execute the 16-scenario simulation matrix and the 4-scenario Raspberry Pi matrix. Record results for F1, AUC, Acc, MSE, MAE, Throughput, and hardware resource utilization.
2. **Preliminary Analysis (Current Week)**: Compare adaptive vs baseline strategies. Validate research hypotheses regarding accuracy-efficiency trade-offs.
3. **Paper Drafting (Next Week)**: Begin drafting the initial manuscript, focusing on the methodology, experiment design, and interpreted findings.
