# Latest stable as of 9 Dec 2025
FROM docker.io/vllm/vllm-openai:v0.12.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TENSOR_PARALLEL=1 \
    ORCH_LOG_LEVEL=INFO \
    ORCH_ENABLE_REQUEST_LOGS=false \
    MODEL_NAME="" \
    ORCH_API_KEY="" \
    ORCH_ENCRYPTION_KEY="" \
    HF_TOKEN="" \
    VLLM_API_KEY="" \
    VLLM_MAX_MODEL_LEN=8192 \
    FINAL_MAX_TOKENS=8192 \
    VLLM_EXTRA_ARGS=""

WORKDIR /app

# Use BuildKit cache mount for apt packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with pip cache mount
# This layer is cached unless requirements.txt changes
COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Copy config files
COPY orchestrator/supervisord.conf /etc/supervisor/conf.d/orchestrator.conf
RUN mkdir -p /app/logs

# Copy static files
COPY orchestrator/static /app/static

# Copy application code last (most frequently changed)
# This layer rebuilds on every code change but is tiny (~1MB)
COPY orchestrator/app /app/orchestrator/app

EXPOSE 8000 8005

# Override base image entrypoint so we own process supervision
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/orchestrator.conf"]

