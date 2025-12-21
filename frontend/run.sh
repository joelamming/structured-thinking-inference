#!/bin/bash
# Start the frontend server

# Load .env file from project root if it exists
if [ -f ../.env ]; then
    echo "Loading environment from project root .env file..."
    export $(grep -v '^#' ../.env | xargs)
elif [ -f .env ]; then
    echo "Loading environment from local .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Check if ORCH_API_KEY is set
if [ -z "$ORCH_API_KEY" ]; then
    echo "⚠️  ERROR: ORCH_API_KEY not set!"
    echo "The orchestrator requires an API key for authentication."
    echo ""
    echo "Set it in your .env file or export it:"
    echo "  export ORCH_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

# Check if ORCH_ENCRYPTION_KEY is set
if [ -z "$ORCH_ENCRYPTION_KEY" ]; then
    echo "⚠️  Warning: ORCH_ENCRYPTION_KEY not set!"
    echo "Messages will NOT be encrypted. This is insecure."
    echo ""
    echo "To generate a key, run:"
    echo "  python3 test_encryption.py"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Default values
ORCHESTRATOR_WS_URL=${ORCHESTRATOR_WS_URL:-"ws://localhost:8005/ws/completions"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-VL-4B-Thinking"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8080}

echo "======================================"
echo "Starting Thinking Chat Frontend"
echo "======================================"
echo "Orchestrator: $ORCHESTRATOR_WS_URL"
echo "Model: $MODEL_NAME"
echo "Server: http://$HOST:$PORT"
echo "API Key: $([ -n "$ORCH_API_KEY" ] && echo "✓ Set" || echo "✗ Missing")"
echo "Encryption: $([ -n "$ORCH_ENCRYPTION_KEY" ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo "======================================"
echo ""

python3 -m uvicorn app:app --host $HOST --port $PORT --reload


