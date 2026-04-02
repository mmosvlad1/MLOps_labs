# Stage 1: builder — install all dependencies with compilers available
FROM python:3.11 AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: runtime — slim image with pre-built packages
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy CLI tools (mlflow, dvc, etc.)
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code
COPY src/ src/
COPY config/ config/
COPY dvc.yaml dvc.lock ./

ENV PYTHONPATH=/app
