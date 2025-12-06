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

### 3) Build and push the orchestrator image (linux/amd64)
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
- Orchestrator: the image you just built.
- vLLM: use upstream `ghcr.io/vllm-project/vllm-openai:latest` (no custom build needed).
- Redis: any standard `redis:alpine` (no persistence required by default).

### 5) Required environment variables (orchestrator)
```
ORCH_API_KEY=...               # required
ORCH_ENCRYPTION_KEY=...        # 32-byte Fernet key, base64; generate via e.g:
# python - <<'PY'
# from cryptography.fernet import Fernet
# print(Fernet.generate_key().decode())
# PY
ORCH_LLM_SERVER_URL=http://vllm:8000/v1/completions
ORCH_LLM_METRICS_URL=http://vllm:8000/metrics
ORCH_EMBEDDING_SERVER_URL=http://vllm:8000/v1/embeddings
ORCH_REDIS_URL=redis://redis:6379/0
ORCH_LOG_LEVEL=INFO
ORCH_ENABLE_REQUEST_LOGS=false
```

### 6) Deploy on RunPod (single container for orchestrator)
- Set container image to your pushed orchestrator tag (e.g., `$REG/orchestrator:latest`).
- If the image is private, add registry auth (PAT/IAM). Or set package to public in GHCR.
- Expose port `8005` (HTTP/WebSocket).
- Provide env vars above. Ensure `ORCH_LLM_*` and `ORCH_REDIS_URL` point to reachable services (either other pods or external endpoints).
- Run vLLM separately as another pod/container using `ghcr.io/vllm-project/vllm-openai:latest` with envs:
  ```
  HF_TOKEN=...
  MODEL_NAME=...
  TENSOR_PARALLEL=1   # adjust for GPU count
  ```
  Expose port `8000`. Point orchestrator URLs at this service.
- Run Redis as another lightweight pod/container (`redis:alpine`) and point `ORCH_REDIS_URL` to it.

### 7) Deploy on ECS (task definition outline)
- Define three containers in one task or separate services:
  - Orchestrator: image `$REG/orchestrator:latest`, port 8005, envs above.
  - vLLM: image `ghcr.io/vllm-project/vllm-openai:latest`, port 8000, envs `HF_TOKEN`, `MODEL_NAME`, `TENSOR_PARALLEL`.
  - Redis: image `redis:alpine`, port 6379.
- Networking: place in the same VPC/security group; give the orchestrator service connectivity to vLLM and Redis via their task/service DNS names.
- If images are in ECR, no extra auth. If GHCR/Docker Hub private, add a registry auth section to the task definition.

### 8) Testing
- Orchestrator health: `GET http://<orchestrator-host>:8005/healthz`
- Dashboard: `http://<orchestrator-host>:8005/dashboard`
- OpenAPI docs: `http://<orchestrator-host>:8005/docs`

### 9) Tips
- Pin images by digest for stability: `$REG/orchestrator@sha256:<digest>`.
- Keep PATs secret; avoid putting them in env vars—use platform registry auth.
- Keep the GHCR package public if you don’t want to manage auth on RunPod.
