## Build and deploy (orchestrator + vLLM) without local Docker on target

This guide shows how to build the orchestrator image on a build box (e.g. EC2), push it to a registry run it on a platform like RunPod or ECS. This avoids Docker-in-Docker on some GPU-as-a-Service vendors.

### Prerequisites
- A Linux/x86_64 build machine with Docker/Buildx (Ubuntu/AL2 is fine; build with `--platform=linux/amd64`).
- A container registry (GHCR, Docker Hub or ECR).
- Git installed.
- Hugging Face token for vLLM (`HF_TOKEN`) and your model name (`MODEL_NAME`), sometimes needed for gated models.

### 1) Clone the repo on the build machine
```bash
git clone https://github.com/joelamming/structured-thinking-inference.git
cd structured-thinking-inference
```

### 2) Log in to your registry
Pick one -- I used GHCR:
- **GHCR (classic PAT with write/read:packages):**
  ```bash
  echo "$PAT" | docker login ghcr.io -u <gh_user> --password-stdin
  export REG=ghcr.io/<gh_user>/structured-thinking-inference
  ```
- **Docker Hub (personal access token or password):**
  ```bash
  echo "$HUB_TOKEN" | docker login -u <hub_user> --password-stdin
  export REG=docker.io/<hub_user>/structured-thinking-inference
  ```
- **ECR (IAM):**
  ```bash
  aws ecr get-login-password --region <region> \
    | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
  export REG=<acct>.dkr.ecr.<region>.amazonaws.com/structured-thinking-inference
  ```

### 3) Build and push the single image (linux/amd64)
```bash
docker buildx create --use --name builder || docker buildx use builder
TAG=$(git rev-parse --short HEAD)
docker buildx build --platform=linux/amd64 \
  -t $REG/orchestrator:latest \
  -t $REG/orchestrator:$TAG \
  -f orchestrator/Dockerfile \
  --push .
```

### 4) Runtime images
- Single container image: the image you just built (contains vLLM server + orchestrator under `supervisord`).

### 5) Required environment variables (orchestrator)
```
ORCH_API_KEY=...               # required
ORCH_ENCRYPTION_KEY=...        # 32-byte Fernet key, base64; generate via e.g:
# python - <<'PY'
# from cryptography.fernet import Fernet
# print(Fernet.generate_key().decode())
# PY
ORCH_LLM_SERVER_URL=http://127.0.0.1:8000/v1/completions
ORCH_LLM_METRICS_URL=http://127.0.0.1:8000/metrics
ORCH_EMBEDDING_SERVER_URL=http://127.0.0.1:8000/v1/embeddings
ORCH_LOG_LEVEL=INFO                   # optional
ORCH_ENABLE_REQUEST_LOGS=false        # optional, write to /app/logs/llm_request_logs.jsonl
MODEL_NAME=...                        # required for vLLM
HF_TOKEN=...                          # only for gated models
TENSOR_PARALLEL=1                     # adjust for GPU count
```

### 6) Deploy on RunPod (single container)
- Set container image to your pushed orchestrator tag (e.g., `$REG/orchestrator:latest`).
- If the image is private, add registry auth (PAT/IAM). Or set package to public in GHCR.
- Expose port `8005` (HTTP/WebSocket).
- Provide env vars above. No external Redis is required; state is in-memory and resets on container restart.
- GPU: enable the RunPod GPU toggle so `vllm serve` can see the device.

### 7) Deploy on ECS (task definition outline)
- Single container in the task:
  - Image `$REG/orchestrator:latest`
  - Port 8005 exposed
  - Env vars above (MODEL_NAME/HF_TOKEN/TENSOR_PARALLEL for vLLM; ORCH_* for FastAPI)
- Assign a GPU to the task definition; no sidecar Redis needed.

### 8) Testing
- Orchestrator health: `GET http://<orchestrator-host>:8005/health`
- Dashboard: `http://<orchestrator-host>:8005/dashboard`
- OpenAPI docs: `http://<orchestrator-host>:8005/docs`
- Completions: WebSocket at `ws://<host>:8005/ws/completions` (final response over WS)

### 9) Tips
- Pin images by digest for stability: `$REG/orchestrator@sha256:<digest>`.
- Keep PATs secret; avoid putting them in env vars—use platform registry auth.
- Keep the GHCR package public if you don’t want to manage auth on RunPod.
