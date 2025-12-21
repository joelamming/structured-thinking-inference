## Build and deploy (orchestrator + vLLM) without local Docker on target

This guide shows how to build the orchestrator image on a build box (e.g. EC2), push it to a registry and run it on platforms like ECS. This avoids Docker-in-Docker quirks on some GPU-as-a-Service vendors.

Note that pulling a 9GB container takes ages. It's often easier to just spin up via the Jupyter Notebook terminal if you're in RunPod. See [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md) for details. The below is best if you plan to cache your containers nearby so they pull fast.

### Prerequisites
- A Linux/x86_64 build machine with Docker/Buildx (Ubuntu/AL2 is fine; build with `--platform=linux/amd64`).
- A container registry (GHCR, Docker Hub or ECR).
- Git installed.
- Hugging Face token for vLLM (`HF_TOKEN`) and model name (`MODEL_NAME`), sometimes needed for gated models.

### 0. Remote into the build machine
```bash
ssh -i <path_to_yer_keys_mate> ec2-user@ec2-<ip-address>.eu-west-2.compute.amazonaws.com
```

### 1. Setup and clone the repo
```bash
# Install docker and git
sudo dnf install -y docker git

# Start docker
sudo systemctl enable --now docker

# Let user run docker without sudo (group created by package)
sudo usermod -aG docker $USER

# close and reconnect

# multi-arch emulation for buildx
docker run --privileged --rm tonistiigi/binfmt --install all

# check stuff
docker --version
docker buildx version
docker run --rm hello-world

git clone https://github.com/joelamming/structured-thinking-inference.git
cd structured-thinking-inference
```

### 2. Log in to your registry
Pick one -- I used GHCR with `gh_user=joelamming`:
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

### 3. Build and push the single image (linux/amd64)
```bash
docker buildx create --use --name builder || docker buildx use builder
TAG=$(git rev-parse --short HEAD)

docker buildx build --platform=linux/amd64 \
  -t $REG:latest \
  -t $REG:$TAG \
  -f orchestrator/Dockerfile \
  --push .
```

> **Note:** First build pulls [~8GB](https://hub.docker.com/layers/vllm/vllm-openai/v0.12.0/images/sha256-f2309d913a07da49ea20b2a694703f4cfcb5ad8e7437ec0f26145479ac01e002) vLLM base image (CUDA + PyTorch + vLLM).

### 4. Runtime images
- Single container image: the image we just built (contains vLLM server + orchestrator under `supervisord`).

### 5. Required environment variables (orchestrator)
```
ORCH_API_KEY=...               # required
ORCH_ENCRYPTION_KEY=...        # 32-byte Fernet key, base64; generate via e.g:
# python3 - <<'PY'
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

### 6. Deploy on CaaS (single container)
- Set container image to your pushed orchestrator tag (e.g., `$REG/orchestrator:latest`).
- If the image is private, add registry auth (PAT/IAM). Or set package to public in GHCR.
- Expose port `8005` (HTTP/WebSocket).
- Provide env vars above; state is in-memory and resets on container restart.
- GPU: enable the CaaS vendor GPU toggle so `vllm serve` can see the device.
- **Recommended:** Mount a persistent volume to `/root/.cache/huggingface` to cache model weights (gigabytes or terabytes depending on your model) across restarts.

### 7. Deploy on ECS (task definition outline)
- Single container in the task:
  - Image `$REG/orchestrator:latest`
  - Port 8005 exposed
  - Env vars above (MODEL_NAME/HF_TOKEN/TENSOR_PARALLEL for vLLM; ORCH_* for FastAPI)

### 8. Testing

#### Local Testing with Docker
```bash
# Create a persistent volume for model cache
docker volume create hf-cache

# Run the container
docker run --rm -it --gpus all \
  -p 8005:8005 -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  -e MODEL_NAME=Qwen/Qwen3-VL-4B-Thinking \
  -e HF_TOKEN=your_token \
  -e ORCH_API_KEY=test-key \
  -e ORCH_ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())") \
  $REG:latest
```

#### Health Checks
- Orchestrator health: `GET http://<orchestrator-host>:8005/health`
- Dashboard: `http://<orchestrator-host>:8005/dashboard`
- OpenAPI docs: `http://<orchestrator-host>:8005/docs`
- Completions: WebSocket at `ws://<host>:8005/ws/completions` (final response over WS)
