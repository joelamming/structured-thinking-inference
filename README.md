# Webserver with configurable CoT with guided decoding final output, inference engine based on vLLM

Public-facing implementation

## Services (single container)

- `vllm` server running inside the same container (port `8000`).
- `orchestrator` FastAPI app (port `8005`) with an in-memory queue/cache (lost on restart).

## Required envs

Export secrets via your runner (e.g. ECS or RunPod env vars) before starting the container:

```
export ORCH_API_KEY=...
export ORCH_ENCRYPTION_KEY=...
export HF_TOKEN=... # usually not needed for public models
export MODEL_NAME=Qwen/Qwen4-420B-A69B-BitNet-Speciale
export ORCH_LOG_LEVEL=INFO
export ORCH_ENABLE_REQUEST_LOGS=false
export TENSOR_PARALLEL=1
```

## Usage

Build/push the single image (see `BUILD_AND_DEPLOY.md` for GHCR instructions), then run it with GPU support. Two processes run under `supervisord` inside the container: `vllm serve` on `8000` and `uvicorn app.main:app` on `8005`.

- WebSocket completions (final response over WS): `ws://localhost:8005/ws/completions`
- Embeddings: `http://localhost:8005/embeddings`
- Dashboard: `http://localhost:8005/dashboard`
- OpenAPI docs: `http://localhost:8005/docs`

For updates:

```
docker compose build orchestrator
docker compose up -d orchestrator
```

Redis data is transient by design; restarts clear pending jobs and metrics snapshots.
