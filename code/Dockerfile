# syntax=docker/dockerfile:1.7

# A lightweight standard Python 3.12 image
FROM python:3.12-slim-bookworm

# Set Python to run unbuffered and quiet warnings
ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venv/bin:${PATH}"

# Working directory inside the container
WORKDIR /app

# Install uv - the fast package manager
RUN pip install uv

# Copy project dependency definitions first for better layer caching.
COPY pyproject.toml uv.lock ./

# Install project dependencies into a venv outside /app so bind mounts do not hide it.
# Force CPU PyTorch backend to avoid pulling CUDA/NVIDIA packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --torch-backend cpu -r pyproject.toml

# Copy proto definitions and compile to Python bindings
COPY proto/ ./proto/
RUN uv pip install --python /opt/venv/bin/python grpcio-tools && \
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto

# Copy the actual Python source code
COPY src/ ./src/
COPY config.yaml /app/config.yaml

# By default, do nothing (docker-compose will override this to start specific nodes)
CMD ["python", "--version"]
