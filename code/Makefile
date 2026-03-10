# FSL Project Makefile
# This file turns complex command lines into simple, one-word commands.

.PHONY: help compile-proto test-network clean

NUM_CLIENTS := $(shell grep "num_clients:" config.yaml | awk '{print $$2}')

help:
	@echo "Available commands:"
	@echo "  make compile-proto   - Re-compiles the fsl.proto file into Python code"
	@echo "  make test-network    - Builds and boots up Server and $(NUM_CLIENTS) Clients"
	@echo "  make clean           - Removes all Docker containers and network caches"

# Compiles the gRPC protocol definitions
compile-proto:
	uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto
	@echo "fsl.proto compiled successfully!"

# Tests the basic Hello World gRPC network connection using Docker Compose
test-network:
	@echo "Starting Split Learning with $(NUM_CLIENTS) clients..."
	docker compose build
# 1. 先背景啟動 Server
	docker compose up -d fsl-server

# 2. 啟動 NUM_CLIENTS 個 Client，docker compose 會自動命名為 fsl-client-1, fsl-client-2, ...
	docker compose up -d --scale fsl-client=$(NUM_CLIENTS) fsl-client

# 3. 顯示目前運行中的容器狀態
	docker compose logs -f

# Cleans up the Docker environment
clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up all FSL Docker containers and networks."
