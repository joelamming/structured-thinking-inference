FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY orchestrator/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip uv && \
    uv pip install --system --no-cache -r /tmp/requirements.txt

COPY orchestrator/app /app/app
COPY orchestrator/static /app/static
RUN mkdir -p /app/logs

EXPOSE 8005

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]

