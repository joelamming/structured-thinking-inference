"""
HTMX-based chat frontend for structured-thinking-inference orchestrator.
Pure HTML/CSS with minimal JS for WebSocket handling.

Architecture:
- Browser <-> Frontend (HTTP/WebSocket - no encryption needed locally)
- Frontend <-> Orchestrator (WebSocket - Fernet encrypted)
"""
import os
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from cryptography.fernet import Fernet
import websockets

app = FastAPI(title="Thinking Chat Frontend")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Load config from env
ENCRYPTION_KEY = os.getenv("ORCH_ENCRYPTION_KEY")
ORCHESTRATOR_WS_URL = os.getenv("ORCHESTRATOR_WS_URL")
ORCHESTRATOR_API_KEY = os.getenv("ORCH_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Validate config
if not ORCHESTRATOR_API_KEY:
    print("⚠️  WARNING: ORCH_API_KEY not set - orchestrator will reject connections!")

# Initialize Fernet cipher for server-to-server encryption
fernet: Optional[Fernet] = None
if ENCRYPTION_KEY:
    try:
        fernet = Fernet(ENCRYPTION_KEY.encode())
        print(f"✓ Encryption enabled with key: {ENCRYPTION_KEY[:20]}...")
    except Exception as e:
        print(f"⚠️  Invalid encryption key: {e}")
        fernet = None
else:
    print("⚠️  ORCH_ENCRYPTION_KEY not set - encryption disabled!")

@app.websocket("/ws")
async def websocket_proxy(websocket: WebSocket):
    """
    Proxy WebSocket connection between browser and orchestrator.
    Handles Fernet encryption/decryption server-side.
    """
    await websocket.accept()
    
    orchestrator_ws = None
    try:
        # Connect to orchestrator with API key authentication
        headers = {}
        if ORCHESTRATOR_API_KEY:
            headers["X-API-Key"] = ORCHESTRATOR_API_KEY
        
        orchestrator_ws = await websockets.connect(
            ORCHESTRATOR_WS_URL,
            additional_headers=headers
        )
        
        async def forward_to_orchestrator():
            """Forward messages from browser to orchestrator (with encryption)"""
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    
                    # Handle pong from browser
                    if data.get("type") == "pong":
                        if orchestrator_ws:
                            await orchestrator_ws.send(json.dumps(data))
                        continue
                    
                    # Encrypt message content if present
                    # Fernet.encrypt() returns a base64-encoded token (as bytes)
                    # Just decode to string for JSON
                    if fernet and "request" in data:
                        for msg in data["request"].get("messages", []):
                            if "content" in msg:
                                # Fernet token is already base64-encoded
                                encrypted_token = fernet.encrypt(msg["content"].encode())
                                msg["content"] = encrypted_token.decode('utf-8')
                    
                    # Forward to orchestrator
                    if orchestrator_ws:
                        await orchestrator_ws.send(json.dumps(data))
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Forward to orchestrator error: {e}")
        
        async def forward_to_browser():
            """Forward messages from orchestrator to browser (with decryption)"""
            try:
                async for message in orchestrator_ws:
                    data = json.loads(message)
                    
                    # Handle ping from orchestrator
                    if data.get("type") == "ping":
                        try:
                            await websocket.send_json(data)
                        except RuntimeError:
                            # WebSocket already closed
                            break
                        continue
                    
                    # Decrypt response content if present
                    # Fernet tokens come as base64-encoded strings from JSON
                    if fernet and data.get("status") == "completed":
                        result = data.get("result", {})
                        for choice in result.get("choices", []):
                            msg = choice.get("message", {})
                            if "content" in msg and msg["content"]:
                                try:
                                    # Fernet token as string, just encode to bytes and decrypt
                                    decrypted = fernet.decrypt(msg["content"].encode()).decode()
                                    msg["content"] = decrypted
                                except Exception as e:
                                    print(f"Failed to decrypt content: {e}")
                                    
                            if "thinking" in choice and choice["thinking"]:
                                try:
                                    # Fernet token as string, just encode to bytes and decrypt
                                    decrypted = fernet.decrypt(choice["thinking"].encode()).decode()
                                    choice["thinking"] = decrypted
                                except Exception as e:
                                    print(f"Failed to decrypt thinking: {e}")
                    
                    # Forward to browser
                    try:
                        await websocket.send_json(data)
                    except RuntimeError:
                        # WebSocket already closed
                        break
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"Forward to browser error: {e}")
        
        # Run both directions concurrently
        await asyncio.gather(
            forward_to_orchestrator(),
            forward_to_browser()
        )
    
    except Exception as e:
        print(f"WebSocket proxy error: {e}")
        try:
            await websocket.send_json({
                "status": "error",
                "error": f"Connection error: {str(e)}"
            })
        except (RuntimeError, Exception):
            # WebSocket already closed, can't send error
            pass
    finally:
        if orchestrator_ws:
            await orchestrator_ws.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main chat interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_name": MODEL_NAME,
        "encryption_enabled": fernet is not None
    })

@app.get("/config")
async def get_config():
    """Return frontend configuration"""
    return {
        "model_name": MODEL_NAME,
        "encryption_enabled": fernet is not None,
        "orchestrator_url": ORCHESTRATOR_WS_URL
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

