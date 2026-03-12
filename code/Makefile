# FSL Project Makefile
# This file turns complex command lines into simple, one-word commands.

.PHONY: help compile-proto run-network test-network clean

NUM_CLIENTS := $(shell grep "num_clients:" config.yaml | awk '{print $$2}')
SERVER_COLOR := \033[1;34m
RESET_COLOR := \033[0m

help:
	@echo "Available commands:"
	@echo "  make compile-proto   - Re-compiles the fsl.proto file into Python code"
	@echo "  make run-network     - Cleans, builds, and starts Server + $(NUM_CLIENTS) Clients"
	@echo "  make test-network    - Backward-compatible alias for run-network"
	@echo "  make clean           - Removes all Docker containers and network caches"

# Compiles the gRPC protocol definitions
compile-proto:
	uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto
	@echo "fsl.proto compiled successfully!"

# Runs a clean split-learning session from scratch
run-network: clean
	@echo "Starting Split Learning with $(NUM_CLIENTS) clients..."
	docker compose build
# 1. 先背景啟動 Server
	docker compose up -d fsl-server

# 2. 逐個啟動 Client，明確注入 CLIENT_ID，避免和 Docker 內部 hostname 混淆
	@for i in $$(seq 1 $(NUM_CLIENTS)); do \
		echo "Starting client $$i..."; \
		docker compose run -d --no-deps --name fsl-client-$$i -e CLIENT_ID=$$i fsl-client; \
	done

# 3. 顯示目前運行中的容器狀態
	@(docker logs -f fsl-server 2>&1 | awk '{printf "$(SERVER_COLOR)%-13s$(RESET_COLOR) | %s\n", "fsl-server", $$0; fflush()}') & \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; \
			1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; \
			3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; \
			5) color="\033[1;37m" ;; \
		esac; \
		(docker logs -f fsl-client-$$i 2>&1 | awk -v prefix=$$(printf "fsl-client-%d" $$i) -v color="$$color" -v reset="$(RESET_COLOR)" '{printf "%s%-13s%s | %s\n", color, prefix, reset, $$0; fflush()}') & \
	done; \
	wait

# Backward-compatible alias
test-network: run-network

# Cleans up the Docker environment
clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up all FSL Docker containers and networks."
