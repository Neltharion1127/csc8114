# FSL Project Makefile
# This file turns complex command lines into simple, one-word commands.

.PHONY: help compile-proto test-network clean

help:
	@echo "Available commands:"
	@echo "  make compile-proto   - Re-compiles the fsl.proto file into Python code"
	@echo "  make test-network    - Boots up the Docker containers to test the Client/Server Ping"
	@echo "  make clean           - Removes all Docker containers and network caches"

# Compiles the gRPC protocol definitions
compile-proto:
	uv run python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto
	@echo "fsl.proto compiled successfully!"

# Tests the basic Hello World gRPC network connection using Docker Compose
test-network:
	docker compose build
	docker compose up --abort-on-container-exit

# Cleans up the Docker environment
clean:
	docker compose down -v --remove-orphans
	@echo "Cleaned up all FSL Docker containers and networks."
