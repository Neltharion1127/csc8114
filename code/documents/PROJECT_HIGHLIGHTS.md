# Project Highlights: Federated Split Learning (FSL) for Rainfall Prediction

This document summarizes the core technologies and implementation highlights of the FSL project, providing developers and researchers with a quick overview of the system architecture and performance optimizations.

## 1. Core Architecture: Federated Split Learning (FSL)
The system combines the advantages of **Federated Learning (FL)** and **Split Learning (SL)**, specifically designed for resource-constrained IoT edge devices:
- **Data Privacy**: Raw weather data stays on the Client; only intermediate features (Smashed Activations) are sent to the cloud.
- **Compute Distribution**: The lightweight `ClientLSTM` is deployed on the edge, while the compute-intensive `ServerHead` (Regressor) resides on the central server.
- **Global Aggregation**: Uses the **FedAvg** algorithm to periodically synchronize weights across all clients for collaborative learning.

## 2. Data Engineering & Preprocessing
Optimizations targeted at common data imbalance and noise in rainfall prediction:
- **Open-Meteo API Integration**: Switched to a stable, high-quality weather API providing precise historical data for 12 sensors in the Newcastle area.
- **50/50 Balanced Sampling**: Forces training batches to have an equal distribution of rainy and dry samples. This prevents the model from "always guessing zero" due to the 90%+ dry background.
- **Dynamic Z-Score Standardization**:
  - Automatically scans assigned sensor data upon client startup.
  - Calculates Mean and Std on-the-fly to scale input features, ensuring faster model convergence.

## 3. Stability & Performance Enhancements
In the `train_v2_optimize` branch, the training pipeline has been significantly hardened:
- **Gradient Clipping**: Uses `max_norm=1.0` to truncate gradients, effectively solving the "Gradient Explosion" problem common in LSTMs.
- **Architecture Upgrades (64 → 32 → 1)**:
  - **Server Activation**: Switched from ReLU to `LeakyReLU(0.1)` to ensure smooth signal flow.
  - **Generalization**: Added `Dropout` layers to the client model to prevent over-fitting.
- **Early Stopping**: Implemented an automated termination mechanism with a 20-round patience to save compute and avoid overtraining.

## 4. Infrastructure & Tooling
- **Containerized Deployment**: Docker Compose-based virtual subnet for one-click simulation of distributed training across multiple clients.
- **High-Efficiency Communication**: Uses **gRPC** for data exchange, paired with efficient serialization to transmit Activations and Gradients.
- **Automated Evaluation**: Added `run_evaluation.py` to automatically match the best Server and Client weights from a session and generate performance reports on a 14-day test set.
- **Visualization Dashboards**: Built-in plotting tools to track Loss curves, rainfall classification accuracy, and communication latency.
