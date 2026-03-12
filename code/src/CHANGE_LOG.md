
# Change Log

## Branch: train_v2_optimize

Mainly adjusted: `split_lstm.py`, `client_node.py`, `server_node.py`

### 1. Model Architecture Upgrade
- **Prevents Overfitting**: Added Dropout layers to the Client LSTM.
- **Smoother Signals**: Switched Server to LeakyReLU to prevent neuron failure.
- **Stable Structure**: Fixed network to 64 → 32 → 1 for consistent feature learning.

### 2. Data Preprocessing
- **Auto-Standardization (Z-score)**: Automatically calculates Mean and Std for data.
- **Data Alignment**: Scales sensor data to the same range so the model learns faster.

### 3. Training Stability
- **Prevents Crashes**: Added Gradient Clipping to stop training from exploding.
- **Steadier Progress**: Lowered learning rate (0.001 → 0.0005) for smoother convergence.
- **Stop When Done**: Added Early Stopping (patience=20) to end training if no progress is made.

### 4. Tools & Documentation
- **One-Click Eval**: Added `run_evaluation.py` to test the best model on a 14-day test set.
- **Updated Guides**: Added visualization and evaluation tutorials to `dev_guide.md`.

### 5. Logs & Partitioning
- **Early Splitting**: Sensors are partitioned at the start for better normalization accuracy.
- **Detailed Logs**: Logs now track latency, payload size, and save backup config files.
