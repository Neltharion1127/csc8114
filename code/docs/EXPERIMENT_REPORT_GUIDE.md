# FSL Rainfall Prediction: Experiment Report Guide

## 1. Model Architecture
### A. Edge Model (Client LSTM)
- **Input**: 24h x 5 meteorological features.
- **Core**: Double-layer LSTM (Hidden Size: 64, num_layers: 2).
- **Dropout**: 0.3.
- **Output**: 64-dimensional Smashed Activation.

### B. Cloud Model (Server Head)
- **Structure**: 2-layer MLP (64 → 32 → 1).
- **Activation**: LeakyReLU(0.1) in middle, ReLU at output.
- **Output**: Future 3h rainfall (mm).

## 2. Report Requirements
### Section 1: Data Engineering
- Describe **Balanced Sampling** (50/50 rain/dry).
- Explain **Dynamic Z-score** normalization.

### Section 2: Training Config
- Optimizer: Adam (LR=0.0005).
- Clipping: `clip_grad_norm_ (1.0)`.
- Stop: Early Stopping (Patience=20).

### Section 3: Metrics
- MSE / MAE.
- Rain/Dry Accuracy.
- Per-sensor consistency.
