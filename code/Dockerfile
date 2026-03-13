# syntax=docker/dockerfile:1.7

# A lightweight standard Python 3.12 image
FROM python:3.12-slim-bookworm

# Set Python to run unbuffered and quiet warnings
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV UV_LINK_MODE=copy
ENV PATH="/opt/venv/bin:${PATH}"

# Working directory inside the container
WORKDIR /app

# Install uv - the fast package manager
RUN pip install uv

# Copy the dependency lockfiles first to cache the layer
COPY pyproject.toml uv.lock ./

# Install all project dependencies into a venv outside /app so bind mounts do not hide it
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Copy the actual Python source code and generated proto files into the image
COPY proto/ ./proto/
COPY src/ ./src/
COPY config.yaml /app/config.yaml

# By default, do nothing (docker-compose will override this to start specific nodes)
CMD ["python", "--version"]

