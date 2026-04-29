# syntax=docker/dockerfile:1.7

# stage 1: builder
FROM python:3.12-slim-bookworm as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python --torch-backend cpu -r pyproject.toml && \
    uv pip install --python /opt/venv/bin/python grpcio-tools

# stage 2: runtime
FROM python:3.12-slim-bookworm as runtime
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

COPY src/ ./src/
COPY config.yaml /app/config.yaml
COPY proto/ ./proto/

# Compile proto after COPY src/ to avoid being overwritten
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto

CMD ["python", "--version"]
