FROM docker.io/vllm/vllm-openai:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TENSOR_PARALLEL=1 \
    ORCH_LOG_LEVEL=INFO \
    ORCH_ENABLE_REQUEST_LOGS=false \
    HF_TOKEN="" \
    VLLM_API_KEY=""

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY orchestrator/app /app/app
COPY orchestrator/static /app/static
COPY orchestrator/supervisord.conf /etc/supervisor/conf.d/orchestrator.conf
RUN mkdir -p /app/logs

EXPOSE 8000 8005

# Override base image entrypoint so we own process supervision
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/orchestrator.conf"]

