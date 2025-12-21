# RunPod Deployment Guide

This is the one working, end-to-end path to run:

- **vLLM** privately on `127.0.0.1:8000`
- **Orchestrator** publicly on `0.0.0.0:8005` (API-key protected)

## RunPod pod setup

- **Base image**: a vanilla CUDA **12.9** image (Ubuntu 24.04 / “noble” is fine)
- **GPU**: any NVIDIA GPU with a driver that reports **CUDA 12.9** via `nvidia-smi`
- **Ports**: expose **`8005` only** (do not expose `8000`)
- **Terminal**: enable whatever gives you a shell (Jupyter terminal / web terminal / SSH)

## Required environment variables

Set these in the RunPod UI (or export them in your shell):

```
HF_TOKEN=hf_xxxxx
ORCH_API_KEY=key_here
ORCH_ENCRYPTION_KEY=key_here
ORCH_LLM_SERVER_URL=http://127.0.0.1:8000/v1/completions
ORCH_LLM_METRICS_URL=http://127.0.0.1:8000/metrics
ORCH_EMBEDDING_SERVER_URL=http://127.0.0.1:8000/v1/embeddings
ORCH_LOG_LEVEL=INFO
ORCH_ENABLE_REQUEST_LOGS=false
MODEL_NAME=Qwen/Qwen3-VL-4B-Thinking
TENSOR_PARALLEL=1
VLLM_MAX_MODEL_LEN=32000
```

## Commands (run in the pod terminal)

### 1) Install OS packages + `uv`

```bash
apt-get update && apt-get install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 2) Clone repo and create venv

```bash
cd /workspace
git clone https://github.com/joelamming/structured-thinking-inference.git
cd structured-thinking-inference
uv venv
```

### 3) Check environment variables

```bash
required_vars=(
  HF_TOKEN
  ORCH_API_KEY
  ORCH_ENCRYPTION_KEY
  ORCH_LLM_SERVER_URL
  ORCH_LLM_METRICS_URL
  ORCH_EMBEDDING_SERVER_URL
  ORCH_LOG_LEVEL
  ORCH_ENABLE_REQUEST_LOGS
  MODEL_NAME
  TENSOR_PARALLEL
  VLLM_MAX_MODEL_LEN
)

echo "Checking environment variables..."
missing_count=0

for v in "${required_vars[@]}"; do
  if [ -z "${!v:-}" ]; then
    echo "❌ Missing: $v"
    missing_count=$((missing_count + 1))
  else
    echo "✅ Found: $v"
  fi
done

if [ $missing_count -gt 0 ]; then
  echo ""
  echo "ERROR: $missing_count variable(s) missing. Please set them before continuing."
  exit 1
fi

echo ""
echo "All required environment variables are set!"
```

### 4) Install Python deps (orchestrator + vLLM)

```bash
uv pip install --python ./.venv/bin/python -r orchestrator/requirements.txt
uv pip install --python ./.venv/bin/python -U "vllm==0.12.0" uvicorn hf_transfer
```

### 5) Validate encryption key

```bash
./.venv/bin/python - <<'PY'
import os
from cryptography.fernet import Fernet
try:
    Fernet(os.environ["ORCH_ENCRYPTION_KEY"].encode())
    print("✅ OK: ORCH_ENCRYPTION_KEY is a valid Fernet key")
except Exception as e:
    print(f"❌ ERROR: Invalid encryption key - {e}")
    exit(1)
PY
```

### 6) Start vLLM (private)

```bash
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export HF_HUB_ENABLE_HF_TRANSFER=1

./.venv/bin/vllm serve "$MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$VLLM_MAX_MODEL_LEN"
```

### 7) Start the orchestrator (public)

Open a second terminal pane and run:

```bash
cd /workspace/structured-thinking-inference
./.venv/bin/uvicorn orchestrator.app.main:app --host 0.0.0.0 --port 8005
```

## URLs

- **Orchestrator dashboard**: `https://<pod-id>-8005.proxy.runpod.net/dashboard`
- **Orchestrator API docs**: `https://<pod-id>-8005.proxy.runpod.net/docs`
- **Health**: `https://<pod-id>-8005.proxy.runpod.net/health`
