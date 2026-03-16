# FSL Project Makefile
# This file turns complex command lines into simple, one-word commands.

.PHONY: help compile-proto download-data run-network test-network run-native run-native-server run-native-client run-native-clients plot plot-latest plot-session plot-confusion eval-latest eval-session eval-batch matrix matrix-dry-run clean-native check-native-port clean clean-all

DEFAULT_NUM_CLIENTS := $(shell grep "num_clients:" config.yaml | awk '{print $$2}')
NUM_CLIENTS ?= $(DEFAULT_NUM_CLIENTS)
SERVER_COLOR := \033[1;34m
RESET_COLOR := \033[0m
SERVER_HOST ?= 127.0.0.1
SERVER_PORT ?= 50051
SERVER_DEVICE ?= cpu
CLIENT_DEVICE ?= mps
STARTUP_TIMEOUT ?= 60
PLOT_DEVICE ?= cpu
AUTO_PLOT ?= 0
UV_CACHE_DIR ?= .uv-cache
UV := UV_CACHE_DIR=$(UV_CACHE_DIR) UV_LINK_MODE=copy uv
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

help:
	@echo "Available commands:"
	@echo "  make compile-proto   - Re-compiles the fsl.proto file into Python code"
	@echo "  make download-data   - Downloads weather data into dataset/processed"
	@echo "  make run-network     - Cleans, builds, and starts Server + $(NUM_CLIENTS) Clients"
	@echo "  make test-network    - Backward-compatible alias for run-network"
	@echo "  make run-native      - Start native server first, then native clients"
	@echo "  make run-native-server   - Start native server (no Docker)"
	@echo "  make run-native-client   - One command to start 1..NUM_CLIENTS native clients"
	@echo "  make plot SESSION=<id> - Manually plot one session"
	@echo "  make plot-latest     - Plot latest session (training curve + server metrics)"
	@echo "  make plot-session SESSION=<id> - Plot a specific session"
	@echo "  make plot-confusion SESSION=<id> - Plot confusion matrices for one session"
	@echo "  AUTO_PLOT=1          - Enable auto-plot after run-native/run-network (default: 0)"
	@echo "  make eval-latest     - Run structured evaluation for latest bestweights session"
	@echo "  make eval-session SESSION=<id> - Run structured evaluation for one session"
	@echo "  make eval-batch [FORCE_THRESHOLD=0.34] [REPORT_TAG=fixedthr034] - Batch run evaluation over sessions"
	@echo "  make matrix [ONLY=M01,M02] [MAX_RUNS=N] [BACKEND=native|docker] - Run experiment matrix from config.yaml"
	@echo "  make matrix-dry-run [ONLY=M01,M02] [MAX_RUNS=N] [BACKEND=native|docker] - Print matrix plan only"
	@echo "  make clean-native    - Kill native server/client Python processes"
	@echo "  make clean           - Removes all Docker containers and network caches"

# Compiles the gRPC protocol definitions
compile-proto:
	$(UV) run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto
	@echo "fsl.proto compiled successfully!"

# Downloads the latest configured weather dataset into dataset/processed
download-data:
	$(UV) run python -m src.data.data_download_openmeteo

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
	wait; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; else echo "Auto-plot disabled (AUTO_PLOT=$(AUTO_PLOT))"; fi

# Backward-compatible alias
test-network: run-network
# Cleans up the Docker environment
clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up all FSL Docker containers and networks."


run-native-server:
	@echo "Starting native server on $(SERVER_HOST):$(SERVER_PORT) with device=$(SERVER_DEVICE)"
	@$(MAKE) check-native-port
	PYTHONUNBUFFERED=1 FSL_DEVICE=$(SERVER_DEVICE) FSL_SERVER_HOST=$(SERVER_HOST) FSL_SERVER_BIND_HOST=$(SERVER_HOST) FSL_SERVER_PORT=$(SERVER_PORT) \
		$(PYTHON) -u -m src.nodes.server_node

run-native-client:
	@echo "Starting $(NUM_CLIENTS) native clients -> $(SERVER_HOST):$(SERVER_PORT), device=$(CLIENT_DEVICE)"
	@pids=""; \
	trap 'kill $$pids 2>/dev/null || true' INT TERM EXIT; \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; \
			1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; \
			3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; \
			5) color="\033[1;37m" ;; \
		esac; \
		( \
			PYTHONUNBUFFERED=1 CLIENT_ID=$$i FSL_DEVICE=$(CLIENT_DEVICE) FSL_SERVER_HOST=$(SERVER_HOST) FSL_SERVER_PORT=$(SERVER_PORT) \
			$(PYTHON) -u -m src.nodes.client_node 2>&1 | \
			awk -v prefix=$$(printf "client-%d" $$i) -v color="$$color" -v reset="$(RESET_COLOR)" '{printf "%s%-13s%s | %s\n", color, prefix, reset, $$0; fflush()}' \
		) & \
		pids="$$pids $$!"; \
	done; \
	wait $$pids; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; else echo "Auto-plot disabled (AUTO_PLOT=$(AUTO_PLOT))"; fi

plot-latest:
	@SESSION=$$(ls -1dt results/20* 2>/dev/null | head -n1 | xargs -I{} basename {}); \
	if [ -z "$$SESSION" ]; then \
		echo "No results session found under results/"; \
		exit 1; \
	fi; \
	echo "Generating plots for session $$SESSION..."; \
	$(PYTHON) -m src.data.plot_training_curve --session $$SESSION --device $(PLOT_DEVICE); \
	$(PYTHON) -m src.data.plot_server_metrics --log results/$$SESSION/server_log_$$SESSION.csv; \
	$(PYTHON) -m src.data.plot_confusion_matrix --session $$SESSION --phase both

plot-session:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make plot-session SESSION=2026-03-13_01-53-07 [PLOT_DEVICE=cpu|mps]"; \
		exit 1; \
	fi
	@echo "Generating plots for session $(SESSION)..."
	@$(PYTHON) -m src.data.plot_training_curve --session $(SESSION) --device $(PLOT_DEVICE)
	@$(PYTHON) -m src.data.plot_server_metrics --log results/$(SESSION)/server_log_$(SESSION).csv
	@$(PYTHON) -m src.data.plot_confusion_matrix --session $(SESSION) --phase both

plot: plot-session

plot-confusion:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make plot-confusion SESSION=2026-03-13_01-53-07"; \
		exit 1; \
	fi
	@echo "Generating confusion matrix for session $(SESSION)..."
	@$(PYTHON) -m src.data.plot_confusion_matrix --session $(SESSION) --phase both

eval-latest:
	@SESSION=$$(ls -1dt bestweights/20* 2>/dev/null | head -n1 | xargs -I{} basename {}); \
	if [ -z "$$SESSION" ]; then \
		echo "No bestweights session found under bestweights/"; \
		exit 1; \
	fi; \
	echo "Running evaluation for session $$SESSION..."; \
	$(PYTHON) -m src.data.run_evaluation --device $(PLOT_DEVICE) --session $$SESSION

eval-session:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make eval-session SESSION=2026-03-13_03-19-17 [PLOT_DEVICE=cpu|mps]"; \
		exit 1; \
	fi
	@echo "Running evaluation for session $(SESSION)..."
	@$(PYTHON) -m src.data.run_evaluation --device $(PLOT_DEVICE) --session $(SESSION)

eval-batch:
	@CMD="$(PYTHON) -m src.data.batch_run_evaluation --sessions-root $(if $(SESSIONS_ROOT),$(SESSIONS_ROOT),bestweights) --device $(PLOT_DEVICE)"; \
	if [ -n "$(ONLY)" ]; then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(LIMIT)" ]; then CMD="$$CMD --limit $(LIMIT)"; fi; \
	if [ -n "$(FORCE_THRESHOLD)" ]; then CMD="$$CMD --force-prob-threshold $(FORCE_THRESHOLD)"; fi; \
	if [ -n "$(REPORT_TAG)" ]; then CMD="$$CMD --report-tag $(REPORT_TAG)"; fi; \
	if [ "$(CONTINUE_ON_ERROR)" = "1" ]; then CMD="$$CMD --continue-on-error"; fi; \
	if [ "$(DRY_RUN)" = "1" ]; then CMD="$$CMD --dry-run"; fi; \
	echo "Running: $$CMD"; \
	eval "$$CMD"

matrix:
	@CMD="$(PYTHON) -m src.data.run_experiment_matrix --config $(if $(MATRIX_CONFIG),$(MATRIX_CONFIG),config.yaml)"; \
	if [ -n "$(ONLY)" ]; then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(BACKEND)" ]; then CMD="$$CMD --backend $(BACKEND)"; fi; \
	if [ -n "$(MAX_RUNS)" ]; then CMD="$$CMD --max-runs $(MAX_RUNS)"; fi; \
	echo "Running: $$CMD"; \
	eval "$$CMD"

matrix-dry-run:
	@CMD="$(PYTHON) -m src.data.run_experiment_matrix --config $(if $(MATRIX_CONFIG),$(MATRIX_CONFIG),config.yaml) --dry-run"; \
	if [ -n "$(ONLY)" ]; then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(BACKEND)" ]; then CMD="$$CMD --backend $(BACKEND)"; fi; \
	if [ -n "$(MAX_RUNS)" ]; then CMD="$$CMD --max-runs $(MAX_RUNS)"; fi; \
	echo "Running: $$CMD"; \
	eval "$$CMD"

run-native-clients: run-native-client

run-native:
	@echo "Starting native stack: server($(SERVER_DEVICE)) + $(NUM_CLIENTS) clients($(CLIENT_DEVICE))"
	@$(MAKE) check-native-port
	@set -e; \
	pids=""; \
	trap 'kill $$pids 2>/dev/null || true' INT TERM EXIT; \
	( \
		PYTHONUNBUFFERED=1 FSL_DEVICE=$(SERVER_DEVICE) FSL_SERVER_HOST=$(SERVER_HOST) FSL_SERVER_BIND_HOST=$(SERVER_HOST) FSL_SERVER_PORT=$(SERVER_PORT) \
		$(PYTHON) -u -m src.nodes.server_node 2>&1 | \
		awk -v color="$(SERVER_COLOR)" -v reset="$(RESET_COLOR)" '{printf "%s%-13s%s | %s\n", color, "server", reset, $$0; fflush()}' \
	) & \
	server_pid=$$!; \
	pids="$$server_pid"; \
	echo "Waiting for server at $(SERVER_HOST):$(SERVER_PORT) (timeout $(STARTUP_TIMEOUT)s)..."; \
	ready=0; \
	for _ in $$(seq 1 $(STARTUP_TIMEOUT)); do \
		if python -c 'import socket,sys; s=socket.socket(); s.settimeout(1); s.connect((sys.argv[1], int(sys.argv[2]))); s.close()' "$(SERVER_HOST)" "$(SERVER_PORT)" >/dev/null 2>&1; then \
			ready=1; \
			break; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$ready" -ne 1 ]; then \
		echo "Server did not become ready in time."; \
		kill $$pids 2>/dev/null || true; \
		exit 1; \
	fi; \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; \
			1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; \
			3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; \
			5) color="\033[1;37m" ;; \
		esac; \
		( \
			PYTHONUNBUFFERED=1 CLIENT_ID=$$i FSL_DEVICE=$(CLIENT_DEVICE) FSL_SERVER_HOST=$(SERVER_HOST) FSL_SERVER_PORT=$(SERVER_PORT) \
			$(PYTHON) -u -m src.nodes.client_node 2>&1 | \
			awk -v prefix=$$(printf "client-%d" $$i) -v color="$$color" -v reset="$(RESET_COLOR)" '{printf "%s%-13s%s | %s\n", color, prefix, reset, $$0; fflush()}' \
		) & \
		pids="$$pids $$!"; \
	done; \
	wait $$pids; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; else echo "Auto-plot disabled (AUTO_PLOT=$(AUTO_PLOT))"; fi

check-native-port:
	@if lsof -iTCP:$(SERVER_PORT) -sTCP:LISTEN -n -P >/dev/null 2>&1; then \
		echo "Port $(SERVER_PORT) already has listener(s). Run 'make clean-native' first."; \
		lsof -iTCP:$(SERVER_PORT) -sTCP:LISTEN -n -P; \
		exit 1; \
	fi

clean-native:
	@pkill -f "src.nodes.client_node" 2>/dev/null || true
	@pkill -f "src.nodes.server_node" 2>/dev/null || true
	@echo "Killed native server/client processes (if any)."
