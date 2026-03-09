# Federated Split Learning (FSL) - Rainfall Prediction

This repository implements a **Federated Split Learning (FSL)** architecture to predict rainfall using real-time IoT weather data from the Newcastle Urban Observatory.

Currently, **Phase 1 (Data Engineering & Pipeline)** is fully implemented. The system fetches live sensor records, cleans and aligns the irregular time series, and structures them into sliding windows suitable for a PyTorch Deep Learning model, ready to be distributed across simulated edge clients.

---

## 🛠 Project Setup

This project uses `uv` for lightning-fast dependency management and virtual environments.

**1. Install dependencies:**
```bash
uv sync
```
*(Dependencies are tracked in `pyproject.toml`, including `pandas`, `scikit-learn`, `pyarrow`, `torch`, and the UO `uo-pyfetch` library).*

---

## 📦 Data Pipeline (Phase 1)

The data pipeline transforms raw, noisy, unbalanced sensor signals into pristine scaled Tensors for machine learning. This involves three sequential scripts:

### 1. Data Download (`src/data/data_download.py`)
Downloads historical weather observations (Rain, Temperature, Humidity, Pressure, Wind Speed) using the Urban Observatory API.
* Handled the API's constraints by fetching the data explicitly backward in 30-day chunks.
* Extracted approx. 180 days of valid historical records.
* Saves the raw, unaligned multi-sensor log as `dataset/newcastle_rainfall_data.csv`.

**Run:**
```bash
uv run src/data/data_download.py
```

### 2. Data Preprocessing (`src/data/data_preprocessing.py`)
Cleans the raw API logs, which are originally in a "long" untidy format with misaligned timestamps and severe dropouts.
* **Filtering:** Removes broken sensors that lack core meteorological features or have less than 100 hours of uptime. (Reduced from 22 to 12 high-quality sensors).
* **Pivoting & Resampling:** Converts Variable-Value pairs into columns and forces a strict **1-hour interval** cadence. Applies `.sum()` for rainfall (accumulated) and `.mean()` for other features.
* **Imputation:** Forward-fills environmental gaps (up to 6 hours) and zero-fills periods of no rain.
* **Storage:** Outputs one highly compressed Columnar Binary `.parquet` file per valid sensor into `dataset/processed/`.

**Run:**
```bash
uv run src/data/data_preprocessing.py
```

### 3. Federated Dataloader (`src/data/dataloader.py`)
Bridges the gap between Parquet data and PyTorch training. 
* **Non-IID Distribution:** Simulates edge-device constraints by splitting the 12 processed sensors among an arbitrary number of clients (e.g., 3 clients receive 4 unique geographical sensors each).
* **Standardization:** Embeds a Scikit-Learn `StandardScaler` to ensure all 5 meteorological features have a zero mean and unit variance, preventing gradient explosion in LSTM layers.
* **Sliding Window:** Generates `(X, y)` extraction on the fly. Yields 3D PyTorch Tensors shaped `(Batch_Size, 24, 5)` representing the past 24 hours of weather, paired with a target tensor shaped `(Batch_Size, 1)` representing the rainfall prediction for the upcoming 1 hour.

**Run / Test Shapes:**
```bash
uv run python src/data/dataloader.py
```

---

## 🧠 Split Architecture (Phase 2)

The neural network is physically partitioned into two distinct components to protect raw edge data:
1. **`ClientLSTM`:** An edge model (`src/models/split_lstm.py`) that compresses 24-hour time series into a dense 64-dimensional sequence (the "Smashed Activation").
2. **`ServerHead`:** A centralized MLP regressor that takes the activation and predicts the next 1-hour rainfall.
* Both models were tested locally in `test_split_forward.py` to verify that `loss.backward()` gradients successfully cross the detached computational split.

---

## 📡 Docker & gRPC Network (Phases 3 & 4)

To simulate true Edge-to-Cloud communication without deploying physical hardware, the system is fully containerized using **Docker Compose** and connected via **gRPC Protocol Buffers**.

* **Protocol Definition (`proto/fsl.proto`)**: Defines the strict `ForwardRequest` and `SyncRequest` byte packets.
* **Serialization (`src/shared/serialization.py`)**: Safely converts 10MB PyTorch Tensors into 9KB binary streams.
* **Virtual Subnet (`docker-compose.yml`)**: Isolates the `fsl-client` and `fsl-server` models into pseudo-distributed machines talking over TCP port `50051`.

### 🛠 Makefile Commands (How to Run)
We use a `Makefile` to simplify checking the complex Docker network status:

```bash
# 1. Compile the .proto file into auto-generated Python routing code (Needed if you edit fsl.proto)
make compile-proto

# 2. Boot up the Docker virtual environment and test the Server / Client Hello World connection
make test-network

# 3. Destroy all FSL containers and cleanup the network
make clean
```

---

## 🚀 Next Steps (Phase 5)
1. **Connect the PyTorch Models to the gRPC Nodes:** Replace the dummy ping-pong strings with actual Smashed Activations and Gradients.
2. **Implement the Distributed Training Loop:** Combine the `DataLoader`, the PyTorch `optimizers`, and the networking layer to start the first full Federated Split Learning epoch.
