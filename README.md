# Docker Compose stack for a CoT-focused guided decoding inference engine

## Services

- `redis`: Ephemeral queue/cache/metrics store (no persistence).
- `orchestrator`: FastAPI app handling encryption, Redis queueing, WebSocket completions, embeddings and the HTMX metrics dashboard.
- `vllm`: Official `ghcr.io/vllm-project/vllm-openai` server exposing the OpenAI-compatible API surface on port `8000`.

## Required envs

Export secrets via cloud secret manager *before* running Compose. Compose v5 will read host env vars such as:

```
export API_KEY=...
export ENCRYPTION_KEY=...
export HF_TOKEN=...
export MODEL_NAME=Qwen/Qwen4-235B-A22B-BitNet-Speciale
export LOG_LEVEL=INFO
export ENABLE_REQUEST_LOGS=false
export TENSOR_PARALLEL=1
```

## Usage

```
docker compose build --pull
docker compose up -d --wait
```

- WebSocket/HTTP entry point: `http://localhost:8005`
- Dashboard: `http://localhost:8005/dashboard`
- OpenAPI docs: `http://localhost:8005/docs`

For updates:

```
docker compose build orchestrator
docker compose up -d orchestrator
```

Redis data is transient by design; restarts clear pending jobs and metrics snapshots.
