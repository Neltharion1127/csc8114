# Project Highlights: Federated Split Learning (FSL) for Rainfall Prediction

This document summarizes the core technologies and implementation highlights of the FSL project.

## 1. Core Architecture: Federated Split Learning (FSL)
- **Data Privacy**: Raw data stays on the Client; only intermediate features (Smashed Activations) are sent to the cloud.
- **Compute Distribution**: Lightweight `ClientLSTM` on the edge, compute-intensive `ServerHead` on the server.
- **Collaboration**: Uses **FedAvg** algorithm for periodic weight synchronization.

## 2. Data Engineering & Preprocessing
- **Open-Meteo API**: Switched to high-quality historical weather data for Newcastle.
- **50/50 Balanced Sampling**: Forces training batches to have equal rainy/dry samples, preventing "always guessing zero" bias.
- **Dynamic Z-Score**: Automatically normalizes features per sensor group on startup.

## 3. Stability & Performance
- **Gradient Clipping**: Uses `max_norm=1.0` to prevent gradient explosion in LSTMs.
- **Architecture**: 2-layer LSTM (Client) + 2-layer MLP (Server, using LeakyReLU).
- **Early Stopping**: Patience=20 to save compute and avoid overtraining.

## 4. Infrastructure
- **gRPC communication**: Efficient serialization for low-latency transmission.
- **Docker Simulation**: Virtual subnet for multi-client training.
- **Evaluation System**: `run_evaluation.py` for automated precision reports on 14-day test sets.
