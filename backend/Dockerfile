# ────────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Plant Disease AI Backend
# Stage 1: builder  — installs Python dependencies
# Stage 2: production — minimal runtime image
# ────────────────────────────────────────────────────────────────────────────────

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=12.1.0

# ── Stage 1: builder ────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

WORKDIR /build

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.1.40

COPY requirements.txt .
RUN uv pip install --system --no-cache --compile-bytecode -r requirements.txt


# ── Stage 2: production ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS production

LABEL maintainer="plant-disease-ai-team"
LABEL description="Plant Disease Detection & Advisory System — API"

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appgroup . .

# Create directories for runtime use
RUN mkdir -p /app/models /app/static /app/logs && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:$PORT/api/v1/health || exit 1

# Startup command — production grade with uvicorn workers
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2 --proxy-headers --forwarded-allow-ips='*'"]
