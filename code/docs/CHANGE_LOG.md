# Change Log

## Branch: train_v2_optimize

### 1. Model & Training
- **Dropout**: Added to Client LSTM to prevent overfitting.
- **LeakyReLU**: Used in Server to improve signal flow.
- **Stabilization**: Implemented Gradient Clipping (1.0).
- **Control**: Added Early Stopping (Patience 20).

### 2. Data & Sampling
- **Balanced Sampling**: Implemented 50/50 rain/dry sampling to fix "zero-prediction" bias.
- **Z-Score**: Added dynamic sensor-level standardization.
- **Early Splitting**: Standardized 14-day test set partitioning.

### 3. Tooling
- **Logs**: Enhanced server/client CSV logging (Latency, Payload, etc.).
- **Evaluation**: Added `run_evaluation.py` for automated metrics reports.
- **Makefile**: Integrated `run-native` and `plot-latest` workflows.
