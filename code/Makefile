# ═══════════════════════════════════════════════════════════════════════════════
# FSL Project Makefile
#
# Command groups:
#   (no prefix)   Setup utilities  — compile-proto, download-data
#   docker-*      Local Docker dev — single-machine, all containers on one host
#   native-*      Local Python dev — no Docker, direct process execution
#   dist-*        Distributed      — VPS server + Raspberry Pi clients
#   eval-*        Evaluation       — test-set metrics and batch reports
#   plot-*        Visualisation    — training curves, confusion matrices
#   matrix*       Experiment matrix— run the 14-scenario ablation suite
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help compile-proto download-data \
        docker-run docker-build docker-clean \
        native-run native-server native-clients native-check-port native-clean \
        dist-build dist-deploy dist-deploy-check dist-start dist-logs \
        dist-sync-config dist-server-restart dist-load-image dist-load-image-local dist-clean-results dist-clean-server \
        dist-restart \
        eval-latest eval-session eval-batch \
        plot-latest plot-session plot-confusion \
        matrix matrix-dry-run

# ─── Variables ────────────────────────────────────────────────────────────────

# Read num_clients straight from config so it stays in sync automatically
DEFAULT_NUM_CLIENTS := $(shell grep "num_clients:" config.yaml | awk '{print $$2}')
NUM_CLIENTS     ?= $(DEFAULT_NUM_CLIENTS)

# Docker image coordinates
REGISTRY        ?= cindyncl26
CLIENT_IMAGE    ?= $(REGISTRY)/fsl-client:latest
IMAGE_TAG       ?= latest           # Override: make dist-start IMAGE_TAG=sha-abc1234

# Remote hosts
VPS_USER        ?= ubuntu
VPS_HOST_DEPLOY ?= 51.254.207.168

# Network
SERVER_HOST     ?= 0.0.0.0
SERVER_PORT     ?= 50051

# Devices (cpu / cuda / mps)
SERVER_DEVICE   ?= cpu
CLIENT_DEVICE   ?= mps

# Timing
STARTUP_TIMEOUT ?= 60               # Seconds to wait for server readiness

# Analysis
PLOT_DEVICE     ?= cpu
AUTO_PLOT       ?= 0                # Set to 1 to auto-plot after a run

# Python / uv
UV_CACHE_DIR    ?= .uv-cache
UV              := UV_CACHE_DIR=$(UV_CACHE_DIR) UV_LINK_MODE=copy uv
PYTHON          ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

# Terminal colours (used in log multiplexer)
SERVER_COLOR    := \033[1;34m
RESET_COLOR     := \033[0m

# ─── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "── Setup ──────────────────────────────────────────────────────"
	@echo "  compile-proto              Recompile fsl.proto → Python stubs"
	@echo "  download-data              Fetch weather data into dataset/processed/"
	@echo ""
	@echo "── Local Docker (docker-*) ─────────────────────────────────────"
	@echo "  docker-run  [NUM_CLIENTS=N]  Build + start server & N clients in Docker"
	@echo "  docker-build                 Build local amd64 image (no push)"
	@echo "  docker-clean                 Stop & remove all FSL containers"
	@echo ""
	@echo "── Local Native (native-*) ─────────────────────────────────────"
	@echo "  native-run  [NUM_CLIENTS=N]  Start server then all clients as Python processes"
	@echo "  native-server                Start only the server process"
	@echo "  native-clients               Start only the client processes (server must be up)"
	@echo "  native-clean                 Kill all native server/client processes"
	@echo ""
	@echo "── Distributed Pi (dist-*) ─────────────────────────────────────"
	@echo "  dist-build         [IMAGE_TAG=x]  Build amd64+arm64 image and push to Docker Hub"
	@echo "  dist-sync-config                  Push config.yaml to VPS + all Pis"
	@echo "  dist-server-restart               SSH → VPS: stop + recreate server container"
	@echo "  dist-load-image                   Save image from VPS → Mac → all Pis (offline)"
	@echo "  dist-load-image-local             Build arm64 image on Mac → push to all Pis (no VPS/DockerHub needed)"
	@echo "  dist-deploy        [IMAGE_TAG=x]  Ansible: deploy image to all Pis (Tailscale)"
	@echo "  dist-deploy-check                 Ansible dry-run: preview changes without applying"
	@echo "  dist-start         [IMAGE_TAG=x]  Full experiment: sync config → restart server → deploy Pis"
	@echo "  dist-logs                         Tail live server logs from VPS"
	@echo "  dist-clean-results                Delete results/ and bestweights/ on all Pis"
	@echo "  dist-clean-server                 Delete results/ and bestweights/ on VPS"
	@echo "  dist-restart                      Stop all + clear all results + fresh start"
	@echo ""
	@echo "── Evaluation (eval-*) ─────────────────────────────────────────"
	@echo "  eval-latest                  Evaluate the most recent checkpoint session"
	@echo "  eval-session SESSION=<id>    Evaluate a specific session"
	@echo "  eval-batch   [FORCE_THRESHOLD=0.34] [REPORT_TAG=tag]  Batch evaluation"
	@echo ""
	@echo "── Plots (plot-*) ──────────────────────────────────────────────"
	@echo "  plot-latest                  Training curve + server metrics for latest session"
	@echo "  plot-session SESSION=<id>    Same for a specific session"
	@echo "  plot-confusion SESSION=<id>  Confusion matrices for a specific session"
	@echo ""
	@echo "── Experiment Matrix (matrix*) ─────────────────────────────────"
	@echo "  matrix       [ONLY=M01,M02] [MAX_RUNS=N]  Run ablation scenarios"
	@echo "  matrix-dry-run               Print the run plan without executing"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Recompile proto/fsl.proto into proto/fsl_pb2*.py — only needed after editing the .proto
compile-proto:
	$(UV) run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto
	@echo "fsl.proto compiled successfully."

# Download 3 years of hourly weather from Open-Meteo for all configured stations
download-data:
	$(UV) run python -m src.data.data_download_openmeteo

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL DOCKER  (docker-*)
# Single-machine simulation: server + N clients all run as Docker containers
# on the same host, communicating over a Docker bridge network.
# ═══════════════════════════════════════════════════════════════════════════════

# Build image for the local host architecture only (no push, for quick local tests)
docker-build:
	docker build -f Dockerfile -t $(CLIENT_IMAGE) .

# Clean slate run: tear down any leftover containers, rebuild, then start everything.
# Server starts first; clients are launched individually so CLIENT_ID is injected
# explicitly rather than derived from the Docker-assigned hostname.
docker-run: docker-clean
	@echo "Starting $(NUM_CLIENTS) clients in Docker..."
	docker compose build
	docker compose up -d fsl-server
	@for i in $$(seq 1 $(NUM_CLIENTS)); do \
		echo "Starting client $$i..."; \
		docker compose run -d --no-deps --name fsl-client-$$i -e CLIENT_ID=$$i fsl-client; \
	done
	@(docker logs -f fsl-server 2>&1 | awk \
		'{printf "$(SERVER_COLOR)%-13s$(RESET_COLOR) | %s\n", "fsl-server", $$0; fflush()}') & \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; 1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; 3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; 5) color="\033[1;37m" ;; \
		esac; \
		(docker logs -f fsl-client-$$i 2>&1 | awk \
			-v p=$$(printf "fsl-client-%d" $$i) -v c="$$color" -v r="$(RESET_COLOR)" \
			'{printf "%s%-13s%s | %s\n", c, p, r, $$0; fflush()}') & \
	done; \
	wait; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; fi

# Stop and remove all FSL containers and Docker networks
docker-clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up all FSL Docker containers and networks."

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL NATIVE  (native-*)
# Run server and clients as plain Python processes on the local machine.
# No Docker overhead — ideal for rapid iteration and debugging.
# ═══════════════════════════════════════════════════════════════════════════════

# Start only the gRPC server; blocks until you Ctrl-C
native-server:
	@echo "Starting native server on $(SERVER_HOST):$(SERVER_PORT) [device=$(SERVER_DEVICE)]"
	@$(MAKE) native-check-port
	PYTHONUNBUFFERED=1 \
	FSL_DEVICE=$(SERVER_DEVICE) \
	FSL_SERVER_HOST=$(SERVER_HOST) \
	FSL_SERVER_BIND_HOST=$(SERVER_HOST) \
	FSL_SERVER_PORT=$(SERVER_PORT) \
		$(PYTHON) -u -m src.nodes.server_node

# Start all clients in parallel (server must already be running)
native-clients:
	@echo "Starting $(NUM_CLIENTS) native clients → $(SERVER_HOST):$(SERVER_PORT) [device=$(CLIENT_DEVICE)]"
	@pids=""; \
	trap 'kill $$pids 2>/dev/null || true' INT TERM EXIT; \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; 1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; 3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; 5) color="\033[1;37m" ;; \
		esac; \
		( \
			PYTHONUNBUFFERED=1 CLIENT_ID=$$i \
			FSL_DEVICE=$(CLIENT_DEVICE) \
			FSL_SERVER_HOST=$(SERVER_HOST) \
			FSL_SERVER_PORT=$(SERVER_PORT) \
			$(PYTHON) -u -m src.nodes.client_node 2>&1 | \
			awk -v p=$$(printf "client-%d" $$i) -v c="$$color" -v r="$(RESET_COLOR)" \
				'{printf "%s%-13s%s | %s\n", c, p, r, $$0; fflush()}' \
		) & \
		pids="$$pids $$!"; \
	done; \
	wait $$pids; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; fi

# One-shot: start server, wait for it to be ready, then start all clients.
# Ctrl-C kills everything cleanly via the trap.
native-run:
	@echo "Starting native stack: server($(SERVER_DEVICE)) + $(NUM_CLIENTS) clients($(CLIENT_DEVICE))"
	@$(MAKE) native-check-port
	@set -e; \
	pids=""; \
	trap 'kill $$pids 2>/dev/null || true' INT TERM EXIT; \
	( \
		PYTHONUNBUFFERED=1 FSL_DEVICE=$(SERVER_DEVICE) \
		FSL_SERVER_HOST=$(SERVER_HOST) FSL_SERVER_BIND_HOST=$(SERVER_HOST) \
		FSL_SERVER_PORT=$(SERVER_PORT) \
		$(PYTHON) -u -m src.nodes.server_node 2>&1 | \
		awk -v c="$(SERVER_COLOR)" -v r="$(RESET_COLOR)" \
			'{printf "%s%-13s%s | %s\n", c, "server", r, $$0; fflush()}' \
	) & \
	server_pid=$$!; pids="$$server_pid"; \
	echo "Waiting for server at $(SERVER_HOST):$(SERVER_PORT) (timeout $(STARTUP_TIMEOUT)s)..."; \
	ready=0; \
	for _ in $$(seq 1 $(STARTUP_TIMEOUT)); do \
		if python -c \
			'import socket,sys; s=socket.socket(); s.settimeout(1); s.connect((sys.argv[1],int(sys.argv[2]))); s.close()' \
			"$(SERVER_HOST)" "$(SERVER_PORT)" >/dev/null 2>&1; then \
			ready=1; break; \
		fi; \
		sleep 1; \
	done; \
	if [ "$$ready" -ne 1 ]; then \
		echo "Server did not become ready in time. Aborting."; \
		kill $$pids 2>/dev/null || true; exit 1; \
	fi; \
	for i in $$(seq 1 $(NUM_CLIENTS)); do \
		case $$((($$i - 1) % 6)) in \
			0) color="\033[1;32m" ;; 1) color="\033[1;33m" ;; \
			2) color="\033[1;35m" ;; 3) color="\033[1;36m" ;; \
			4) color="\033[1;31m" ;; 5) color="\033[1;37m" ;; \
		esac; \
		( \
			PYTHONUNBUFFERED=1 CLIENT_ID=$$i \
			FSL_DEVICE=$(CLIENT_DEVICE) \
			FSL_SERVER_HOST=$(SERVER_HOST) \
			FSL_SERVER_PORT=$(SERVER_PORT) \
			$(PYTHON) -u -m src.nodes.client_node 2>&1 | \
			awk -v p=$$(printf "client-%d" $$i) -v c="$$color" -v r="$(RESET_COLOR)" \
				'{printf "%s%-13s%s | %s\n", c, p, r, $$0; fflush()}' \
		) & \
		pids="$$pids $$!"; \
	done; \
	wait $$pids; \
	if [ "$(AUTO_PLOT)" = "1" ]; then $(MAKE) plot-latest; fi

# Guard: fail early if the server port is already occupied
native-check-port:
	@if lsof -iTCP:$(SERVER_PORT) -sTCP:LISTEN -n -P >/dev/null 2>&1; then \
		echo "Port $(SERVER_PORT) already in use. Run 'make native-clean' first."; \
		lsof -iTCP:$(SERVER_PORT) -sTCP:LISTEN -n -P; \
		exit 1; \
	fi

# Kill any lingering native server/client Python processes
native-clean:
	@pkill -f "src.nodes.client_node" 2>/dev/null || true
	@pkill -f "src.nodes.server_node" 2>/dev/null || true
	@echo "Native processes stopped."

# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED  (dist-*)
# Real deployment: server on a remote VPS, clients on Raspberry Pis over
# a Tailscale overlay network.
# ═══════════════════════════════════════════════════════════════════════════════

# Sync config.yaml to VPS (scp) and all Pis (ansible).
# Run this whenever config.yaml changes — no need to rebuild the image.
dist-sync-config:
	@echo "=== Syncing config.yaml to VPS ==="
	scp config.yaml $(VPS_USER)@$(VPS_HOST_DEPLOY):~/csc8114/code/config.yaml
	@echo "=== Syncing config.yaml to all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m copy \
	  -a "src=config.yaml dest=/home/pi/config.yaml" \
	  --become

# Restart only the server container on VPS using docker compose.
# Useful after a config change without touching Pi clients.
dist-server-restart:
	@echo "=== Restarting server on VPS ==="
	ssh $(VPS_USER)@$(VPS_HOST_DEPLOY) "\
	  cd ~/csc8114/code && \
	  docker compose -f docker-compose.server.yml down && \
	  docker compose -f docker-compose.server.yml up -d && \
	  echo '[VPS] Server restarted.'"

# Load image onto all Pis without Docker Hub access.
# Saves image from VPS → Mac → pushes to all Pis via Ansible.
# Pi → Docker Hub is often blocked; this works over Tailscale.
dist-load-image:
	@echo "=== [1/3] Saving image from VPS to Mac ==="
	ssh $(VPS_USER)@$(VPS_HOST_DEPLOY) "docker save $(CLIENT_IMAGE) | gzip > /tmp/fsl-client.tar.gz"
	scp $(VPS_USER)@$(VPS_HOST_DEPLOY):/tmp/fsl-client.tar.gz /tmp/fsl-client.tar.gz
	@echo "=== [2/3] Copying image to all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m copy \
	  -a "src=/tmp/fsl-client.tar.gz dest=/tmp/fsl-client.tar.gz" \
	  --become
	@echo "=== [3/3] Loading image on all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m shell \
	  -a "docker load -i /tmp/fsl-client.tar.gz" \
	  --become

# Build arm64 image locally on Mac and push directly to all Pis.
# No Docker Hub or VPS needed — useful when Pi → internet is blocked.
dist-load-image-local:
	@echo "=== [1/3] Building arm64 image on Mac ==="
	docker buildx build --platform linux/arm64 -t fsl-client:arm64 --load -f Dockerfile .
	docker save fsl-client:arm64 | gzip > /tmp/fsl-client.tar.gz
	@echo "=== [2/3] Copying image to all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m copy \
	  -a "src=/tmp/fsl-client.tar.gz dest=/tmp/fsl-client.tar.gz" \
	  --become
	@echo "=== [3/3] Loading image on all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m shell \
	  -a "docker load -i /tmp/fsl-client.tar.gz && docker tag fsl-client:arm64 cindyncl26/fsl-client:latest" \
	  --become

# Delete results/ and bestweights/ on all Pis, then recreate empty dirs.
dist-clean-results:
	ansible clients -i ansible/inventory.taisale.ini \
	  -m shell \
	  -a "rm -rf /home/pi/results /home/pi/bestweights && mkdir -p /home/pi/results /home/pi/bestweights" \
	  --become

# Delete results/ and bestweights/ on the VPS server, then recreate empty dirs.
dist-clean-server:
	ssh $(VPS_USER)@$(VPS_HOST_DEPLOY) "\
	  sudo rm -rf ~/csc8114/code/results ~/csc8114/code/bestweights && \
	  mkdir -p ~/csc8114/code/results ~/csc8114/code/bestweights && \
	  echo '[VPS] results/ and bestweights/ cleared.'"

# Build a multi-architecture image (amd64 for VPS + arm64 for Pi) and push it
# to Docker Hub. Requires `docker buildx` with a multi-platform builder set up.
# Override IMAGE_TAG to version-pin the build, e.g. make dist-build IMAGE_TAG=sha-abc1234
dist-build:
	docker buildx build \
	  --no-cache \
	  --platform linux/amd64,linux/arm64 \
	  -f Dockerfile \
	  -t $(REGISTRY)/fsl-client:$(IMAGE_TAG) \
	  --push .

# Deploy the specified image to all 11 Pis via Ansible over Tailscale.
# Uses inventory.taisale.ini (100.x.x.x Tailscale IPs).
dist-deploy:
	ansible-playbook ansible/deploy_client.yml \
	  -i ansible/inventory.taisale.ini \
	  --extra-vars "image_tag=$(IMAGE_TAG)"

# Dry-run: show what Ansible would change without touching any Pi
dist-deploy-check:
	ansible-playbook ansible/deploy_client.yml \
	  -i ansible/inventory.taisale.ini \
	  --extra-vars "image_tag=$(IMAGE_TAG)" \
	  --check

# Full experiment launch in four steps:
#   1. Sync config.yaml to VPS + all Pis
#   2. SSH into VPS → restart server via docker compose
#   3. Poll VPS_HOST_DEPLOY:SERVER_PORT until gRPC port is open (or timeout)
#   4. Ansible: restart client containers on all Pis
#
# Usage:
#   make dist-start                        # uses IMAGE_TAG=latest
#   make dist-start IMAGE_TAG=sha-a1b2c3   # pins a specific build
dist-start:
	@echo "=== [1/4] Syncing config.yaml to VPS + Pis ==="
	$(MAKE) dist-sync-config
	@echo "=== [2/4] Restarting server on VPS ==="
	$(MAKE) dist-server-restart
	@echo "=== [3/4] Waiting for server at $(VPS_HOST_DEPLOY):$(SERVER_PORT) (timeout=$(STARTUP_TIMEOUT)s) ==="
	@ready=0; \
	for i in $$(seq 1 $(STARTUP_TIMEOUT)); do \
		if python -c \
			'import socket,sys; s=socket.socket(); s.settimeout(1); s.connect((sys.argv[1],int(sys.argv[2]))); s.close()' \
			"$(VPS_HOST_DEPLOY)" "$(SERVER_PORT)" >/dev/null 2>&1; then \
			ready=1; break; \
		fi; \
		printf "."; sleep 1; \
	done; printf "\n"; \
	if [ "$$ready" -ne 1 ]; then \
		echo "ERROR: Server not reachable after $(STARTUP_TIMEOUT)s. Aborting."; \
		exit 1; \
	fi; \
	echo "[OK] Server is reachable."
	@echo "=== [4/4] Deploying clients to Pis (image=$(IMAGE_TAG)) ==="
	ansible-playbook ansible/deploy_client.yml \
	  -i ansible/inventory.taisale.ini \
	  --extra-vars "image_tag=$(IMAGE_TAG)"
	@echo ""
	@echo "Experiment is running. Follow logs with: make dist-logs"

# Nuclear restart: stop everything, wipe all results, then do a full fresh start.
# Equivalent to: stop server + stop all Pi clients + clean VPS + clean Pis + dist-start
dist-restart:
	@echo "=== [1/5] Stopping server on VPS ==="
	ssh $(VPS_USER)@$(VPS_HOST_DEPLOY) "\
	  cd ~/csc8114/code && \
	  docker compose -f docker-compose.server.yml down 2>/dev/null || true && \
	  echo '[VPS] Server stopped.'"
	@echo "=== [2/5] Stopping client containers on all Pis ==="
	ansible clients -i ansible/inventory.taisale.ini \
	  -m shell \
	  -a "docker stop fsl-client 2>/dev/null || true && docker rm fsl-client 2>/dev/null || true" \
	  --become
	@echo "=== [3/5] Clearing results on VPS ==="
	$(MAKE) dist-clean-server
	@echo "=== [4/5] Clearing results on all Pis ==="
	$(MAKE) dist-clean-results
	@echo "=== [5/5] Fresh start ==="
	$(MAKE) dist-start

# Stream live server logs from the VPS; Ctrl-C to stop
dist-logs:
	ssh $(VPS_USER)@$(VPS_HOST_DEPLOY) "docker logs -f fsl-server"

# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (eval-*)
# ═══════════════════════════════════════════════════════════════════════════════

# Evaluate the most recently saved checkpoint against the held-out test set
eval-latest:
	@SESSION=$$(ls -1dt bestweights/20* 2>/dev/null | head -n1 | xargs -I{} basename {}); \
	if [ -z "$$SESSION" ]; then \
		echo "No session found under bestweights/"; exit 1; \
	fi; \
	echo "Evaluating session $$SESSION..."; \
	$(PYTHON) -m src.data.run_evaluation --device $(PLOT_DEVICE) --session $$SESSION

eval-session:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make eval-session SESSION=2026-03-13_03-19-17 [PLOT_DEVICE=cpu|mps]"; exit 1; \
	fi
	$(PYTHON) -m src.data.run_evaluation --device $(PLOT_DEVICE) --session $(SESSION)

# Batch evaluation over multiple sessions; supports optional filters
eval-batch:
	@CMD="$(PYTHON) -m src.data.batch_run_evaluation \
		--sessions-root $(if $(SESSIONS_ROOT),$(SESSIONS_ROOT),bestweights) \
		--device $(PLOT_DEVICE)"; \
	if [ -n "$(ONLY)" ];             then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(LIMIT)" ];            then CMD="$$CMD --limit $(LIMIT)"; fi; \
	if [ -n "$(FORCE_THRESHOLD)" ];  then CMD="$$CMD --force-prob-threshold $(FORCE_THRESHOLD)"; fi; \
	if [ -n "$(REPORT_TAG)" ];       then CMD="$$CMD --report-tag $(REPORT_TAG)"; fi; \
	if [ "$(CONTINUE_ON_ERROR)" = "1" ]; then CMD="$$CMD --continue-on-error"; fi; \
	if [ "$(DRY_RUN)" = "1" ];       then CMD="$$CMD --dry-run"; fi; \
	echo "Running: $$CMD"; eval "$$CMD"

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS  (plot-*)
# ═══════════════════════════════════════════════════════════════════════════════

plot-latest:
	@SESSION=$$(ls -1dt results/20* 2>/dev/null | head -n1 | xargs -I{} basename {}); \
	if [ -z "$$SESSION" ]; then \
		echo "No session found under results/"; exit 1; \
	fi; \
	echo "Plotting session $$SESSION..."; \
	$(PYTHON) -m src.data.plot_training_curve --session $$SESSION --device $(PLOT_DEVICE); \
	$(PYTHON) -m src.data.plot_server_metrics --log results/$$SESSION/server_log_$$SESSION.csv; \
	$(PYTHON) -m src.data.plot_confusion_matrix --session $$SESSION --phase both

plot-session:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make plot-session SESSION=2026-03-13_01-53-07 [PLOT_DEVICE=cpu|mps]"; exit 1; \
	fi
	$(PYTHON) -m src.data.plot_training_curve --session $(SESSION) --device $(PLOT_DEVICE)
	$(PYTHON) -m src.data.plot_server_metrics --log results/$(SESSION)/server_log_$(SESSION).csv
	$(PYTHON) -m src.data.plot_confusion_matrix --session $(SESSION) --phase both

plot-confusion:
	@if [ -z "$(SESSION)" ]; then \
		echo "Usage: make plot-confusion SESSION=2026-03-13_01-53-07"; exit 1; \
	fi
	$(PYTHON) -m src.data.plot_confusion_matrix --session $(SESSION) --phase both

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT MATRIX  (matrix*)
# Runs the 14-scenario ablation suite defined in config.yaml.
# ═══════════════════════════════════════════════════════════════════════════════

matrix:
	@CMD="$(PYTHON) -m src.data.run_experiment_matrix \
		--config $(if $(MATRIX_CONFIG),$(MATRIX_CONFIG),config.yaml)"; \
	if [ -n "$(ONLY)" ];    then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(BACKEND)" ]; then CMD="$$CMD --backend $(BACKEND)"; fi; \
	if [ -n "$(MAX_RUNS)" ]; then CMD="$$CMD --max-runs $(MAX_RUNS)"; fi; \
	echo "Running: $$CMD"; eval "$$CMD"

matrix-dry-run:
	@CMD="$(PYTHON) -m src.data.run_experiment_matrix \
		--config $(if $(MATRIX_CONFIG),$(MATRIX_CONFIG),config.yaml) --dry-run"; \
	if [ -n "$(ONLY)" ];    then CMD="$$CMD --only $(ONLY)"; fi; \
	if [ -n "$(BACKEND)" ]; then CMD="$$CMD --backend $(BACKEND)"; fi; \
	if [ -n "$(MAX_RUNS)" ]; then CMD="$$CMD --max-runs $(MAX_RUNS)"; fi; \
	echo "Running: $$CMD"; eval "$$CMD"
