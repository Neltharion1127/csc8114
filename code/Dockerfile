# syntax=docker/dockerfile:1.7

# stage 1: builder
FROM python:3.12-slim-bookworm as builder
WORKDIR /app
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

COPY proto/ ./proto/
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/fsl.proto

COPY src/ ./src/
COPY config.yaml /app/config.yaml

CMD ["python", "--version"]
