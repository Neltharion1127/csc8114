# A lightweight standard Python 3.12 image
FROM python:3.12-slim-bookworm

# Set Python to run unbuffered and quiet warnings
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

# Working directory inside the container
WORKDIR /app

# Install uv – the fast package manager
RUN pip install uv

# Copy the dependency lockfiles first to cache the layer
COPY pyproject.toml uv.lock ./

# Install all project dependencies into the managed .venv environment
RUN uv sync

# Copy the actual Python source code and generated proto files into the image
COPY proto/ ./proto/
COPY src/ ./src/
COPY config.yaml /app/config.yaml

# By default, do nothing (docker-compose will override this to start specific nodes)
CMD ["python", "--version"]
