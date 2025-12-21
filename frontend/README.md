# Frontend Chat Interface

Shitty vibecoded FastAPI server that proxies WebSocket connections between your browser and the orchestrator.

## Architecture

```
Browser <--WebSocket--> Frontend Server <--WebSocket+Fernet--> Orchestrator
```

The frontend server handles Fernet encryption/decryption so the browser doesn't have to.

## Setup

```bash
pip install -r requirements.txt

export ORCH_ENCRYPTION_KEY="you-wot-m8"
export ORCHESTRATOR_WS_URL="ws://sketchy-hosting-r-us:8005/ws/completions"

./run.sh
```

Open `http://localhost:8080`

## Files

- `app.py` - FastAPI server with WebSocket proxy
- `static/script.js` - Browser WebSocket client
- `static/style.css` - Dark theme
- `static/htmx.min.js` - HTMX library (need to actually use but Claude doesn't know this yet)
