# ── Stage 1: Builder ──────────────────────────────────────────────────────────
# Install dependencies in a separate stage to keep the final image lean.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools
RUN pip install --upgrade pip

# Copy only dependency files first (layer caching — rebuilds only when deps change)
COPY pyproject.toml .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ".[dev]" --no-deps || true
RUN pip install --no-cache-dir \
    timm \
    fastapi \
    uvicorn \
    python-multipart \
    pydantic \
    loguru \
    Pillow \
    numpy


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY medflow/ ./medflow/
COPY configs/ ./configs/

# Non-root user for security
RUN useradd --create-home appuser
USER appuser

# Model checkpoint mounted at runtime (not baked into image)
ENV CHECKPOINT_PATH=/app/checkpoints/best_model.pt
ENV MODEL_VERSION=0.1.0

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "medflow.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
