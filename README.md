# vLLM-based webserver with configurable CoT effort and guided decoding

Public-facing implementation

## Two-pass thinking + guided decoding

- Requests carry `thinking_effort` (`none/low/medium/high`) to size the thinking budget, final answers use a separate fixed token cap.
- We always format the prompt with `<think>` for Qwen-style reasoning. For `none`, we prefill an empty think and skip extra thinking. Other model chat templates can be implemented by inspecting their [`chat_template.json`](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking/blob/main/chat_template.json)
- Step 1: run a thinking call (unless `none`) with a cap -- if it overruns without `</think>` we append a brief cutoff notice.
- Step 2: first guided decode with `structured_outputs` to produce a draft structured answer.
- Step 3: build a correction think that embeds the draft plus "Wait, have I made a mistake?" and run another thinking pass with the same budget (for high effort this can be looong!).
- Step 4: final guided decode, again with `structured_outputs`, using the full combined thinking so the model can revise and still satisfy the schema/regex/choice/grammar.
- This flow maximises adherence to structured outputs while giving the model a chance to self-correct.

### Thinking effort budgets (max tokens)

| Effort | Thinking Budget |
|--------|-----------------|
| `none` | 0 (skip thinking) |
| `low` | 500 |
| `medium` | 2000 |
| `high` | 16000 |

## Architecture

This repo contains two components:

### Orchestrator (runs somewhere with a GPU)
- `vllm` server running inside the same container (port `8000`)
- `orchestrator` FastAPI app (port `8005`) with an in-memory queue/cache (lost on restart)
- Handles WebSocket connections and manages thinking inference
- See `orchestrator/` directory

### Frontend (runs locally)
- Pure HTML/CSS/HTMX chat interface
- Connects to orchestrator via WebSocket
- Fernet encryption for message security
- No frameworks, no build tools, just FastAPI serving HTML
- See `frontend/` directory and `frontend/README.md`

## Required envs

Export secrets via your runner (e.g. ECS or RunPod env vars) before starting the container:

```bash
export ORCH_API_KEY=...
export ORCH_ENCRYPTION_KEY=...      # Fernet key for message encryption
export FINAL_MAX_TOKENS=2048        # Max tokens for final structured output
export HF_TOKEN=...                 # Usually not needed for public models
export MODEL_NAME=Qwen/Qwen4-420B-A69B-BitNet-Speciale
export ORCH_LOG_LEVEL=INFO
export ORCH_ENABLE_REQUEST_LOGS=false
export TENSOR_PARALLEL=1
```

## Usage

Pull the package from the GitHub CR:

```bash
docker pull ghcr.io/joelamming/structured-thinking-inference:latest
```

Or: Build/push the single image (see `BUILD_AND_DEPLOY.md` for GHCR instructions), then run it with GPU support. Two processes run under `supervisord` inside the container: `vllm serve` on `8000` and `uvicorn app.main:app` on `8005`.

### Endpoints

- **WebSocket completions**: `ws://localhost:8005/ws/completions`
- **Embeddings**: `POST http://localhost:8005/embeddings`
- **Dashboard**: `http://localhost:8005/dashboard`
- **OpenAPI docs**: `http://localhost:8005/docs`
- **Health check**: `GET http://localhost:8005/health`

## WebSocket protocol

### Request format

```json
{
  "request": {
    "model": "Qwen/Qwen3-VL-4B-Thinking",
    "messages": [{"role": "user", "content": "<base64-fernet-encrypted>"}],
    "thinking_effort": "medium",
    "response_format": {"type": "json_object"}
  },
  "client_request_id": "optional-tracking-id",
  "timeout_seconds": 120
}
```

### Keepalive (important!)

The server sends periodic pings to keep connections alive during long-running CoT jobs:

```json
{"type": "ping"}
```

**Clients must respond with a pong** to avoid disconnection:

```json
{"type": "pong"}
```

Connections are closed after 45 seconds of inactivity (no messages received).

### Response types

| Type | Description |
|------|-------------|
| `{"type": "ping"}` | Keepalive ping (respond with pong) |
| `{"status": "completed", "result": {...}}` | Successful completion |
| `{"status": "error", "error": "..."}` | Processing error |
| `{"status": "timeout", "error": "..."}` | Request timed out |

### Encryption

All message content is encrypted with Fernet symmetric encryption. Clients must:
1. Encrypt `messages[].content` with the shared `ORCH_ENCRYPTION_KEY` before sending
2. Decrypt `result.choices[].message.content` and `.thinking` from responses

## Frontend quickstart

```bash
cd frontend
pip install -r requirements.txt

# Set your encryption key (must match orchestrator)
export ORCH_ENCRYPTION_KEY="fernet-key"
export ORCHESTRATOR_WS_URL="ws://your-orchestrator:8005/ws/completions"

./run.sh
# Open http://localhost:8080
```

See `frontend/QUICKSTART.md` for details.
