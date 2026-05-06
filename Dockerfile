# ──────────────────────────────────────────────────────────────────────────────
#  Titanic ML v2  —  Multi-stage Dockerfile
#
#  Stage 1 (builder): install deps + train model
#  Stage 2 (runtime): lean production image with pre-trained model baked in
#
#  Key fix from v1: CMD uses $PORT env var (required by Render / Railway / ECS)
# ──────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies (cached layer — only re-runs when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Train the model — baked into the image so the runtime container is self-contained
# (No external storage needed at startup)
RUN python src/train.py


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Only runtime deps (no test packages)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      fastapi uvicorn[standard] pydantic \
      scikit-learn pandas numpy joblib

# Copy source + trained model artefacts from builder
COPY --from=builder /app/src      ./src
COPY --from=builder /app/api      ./api
COPY --from=builder /app/data     ./data
COPY --from=builder /app/model    ./model

# ── Security: non-root user ───────────────────────────────────────────────────
RUN useradd -m appuser
USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" \
  || exit 1

# ── Start API ─────────────────────────────────────────────────────────────────
# $PORT is set by Render / Railway / ECS at runtime  (fixes v1 hardcoded port bug)
CMD uvicorn api.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 2 \
    --log-level info
