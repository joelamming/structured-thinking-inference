"""
HTMX-based chat frontend for structured-thinking-inference orchestrator.
Server renders HTML partials. Minimal client-side JS.

Architecture:
- Browser <-> Frontend (HTTP/WebSocket - HTMX handles UI)
- Frontend <-> Orchestrator (WebSocket - Fernet encrypted)
"""

import asyncio
import json
import os
import re
import time
from typing import Any
from uuid import uuid4

import websockets  # type: ignore[import-untyped]
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load environment variables from .env file
load_dotenv()

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
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"

# Validate config
if MOCK_MODE:
    print("🎭 MOCK MODE ENABLED - No GPU credits will be used!")
    print("   Set MOCK_MODE=false to connect to real orchestrator")
elif not ORCHESTRATOR_API_KEY:
    print("⚠️  WARNING: ORCH_API_KEY not set - orchestrator will reject connections!")

# Initialize Fernet cipher for server-to-server encryption
fernet: Fernet | None = None
if ENCRYPTION_KEY:
    try:
        fernet = Fernet(ENCRYPTION_KEY.encode())
        print(f"✓ Encryption enabled with key: {ENCRYPTION_KEY[:20]}...")
    except Exception as e:
        print(f"⚠️  Invalid encryption key: {e}")
        fernet = None
else:
    print("⚠️  ORCH_ENCRYPTION_KEY not set - encryption disabled!")

# In-memory session store: session_id -> conversation history
sessions: dict[str, list[dict]] = {}


def format_content(text: str) -> str:
    """Format text with markdown-like syntax"""
    # Escape HTML first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Convert ```code blocks``` to <pre><code>
    text = re.sub(r"```(\w+)?\n(.*?)```", r"<pre><code>\2</code></pre>", text, flags=re.DOTALL)

    # Convert `inline code` to <code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Convert **bold** to <strong>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

    # Convert newlines to <br>
    text = text.replace("\n", "<br>")

    return text


def generate_mock_response(user_message: str, effort: str) -> dict[str, Any]:
    """
    Generate a mock orchestrator response without hitting the real backend.
    Matches the exact structure from orchestrator/app/main.py line 129.
    """
    # Mock thinking based on effort
    thinking_samples = {
        "none": "",
        "low": 'Let me think about this quickly...\n\nThe user asked: "' + user_message[:50] + '"',
        "medium": "Alright, let me carefully consider this question.\n\n"
        + 'The user is asking about: "'
        + user_message[:50]
        + '"\n\n'
        + "I should provide a helpful and accurate response. Let me structure my thoughts:\n"
        + "1. First, I'll acknowledge their question\n"
        + "2. Then provide relevant information\n"
        + "3. Finally, offer any additional context that might help",
        "high": "This is an interesting question that deserves careful thought.\n\n"
        + 'Let me break down what the user is asking: "'
        + user_message[:100]
        + '"\n\n'
        + "Key considerations:\n"
        + "- The context and intent behind the question\n"
        + "- What information would be most helpful\n"
        + "- How to structure a clear and comprehensive response\n\n"
        + "After analyzing this thoroughly, I believe the best approach is to...\n"
        + "Actually, wait - let me reconsider. Perhaps a different angle would be better.\n\n"
        + "My final output will therefore be a balanced response that addresses the core question "
        + "while providing useful context and examples.",
    }

    thinking_text = thinking_samples.get(effort, thinking_samples["medium"])

    # Mock response content
    content = (
        f"**[MOCK MODE - No GPU credits used]**\n\n"
        f'I received your message: "{user_message[:100]}..."\n\n'
        f"This is a simulated response. In production, the real model would process your request "
        f"with **{effort}** thinking effort.\n\n"
        f"To connect to the real backend, set `MOCK_MODE=false` in your environment."
    )

    # Encrypt if encryption is enabled (to match real behavior)
    if fernet:
        encrypted_content = fernet.encrypt(content.encode()).decode("utf-8")
        encrypted_thinking = (
            fernet.encrypt(thinking_text.encode()).decode("utf-8") if thinking_text else ""
        )
    else:
        encrypted_content = content
        encrypted_thinking = thinking_text

    # Match exact structure from orchestrator
    return {
        "status": "completed",
        "request_id": f"mock-{int(time.time() * 1000)}",
        "result": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": encrypted_content,
                        "thinking": encrypted_thinking,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        },
    }


@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    """
    HTMX WebSocket handler.
    - Receives HTMX messages (user input + session info)
    - Manages conversation state server-side
    - Sends HTML fragments back to browser
    """
    await websocket.accept()
    session_id: str | None = None
    orchestrator_ws = None
    ping_task = None

    try:
        # Send connected status
        status_html = templates.get_template("status_update.html").render(
            status_class="connected",
            status_text="Connected (Mock Mode)" if MOCK_MODE else "Connected",
        )
        await websocket.send_text(status_html)

        # Connect to orchestrator only if not in mock mode
        if not MOCK_MODE:
            if not ORCHESTRATOR_WS_URL:
                raise ValueError("ORCHESTRATOR_WS_URL not set")

            headers = {}
            if ORCHESTRATOR_API_KEY:
                headers["X-API-Key"] = ORCHESTRATOR_API_KEY

            orchestrator_ws = await websockets.connect(  # type: ignore[attr-defined]
                ORCHESTRATOR_WS_URL, additional_headers=headers
            )

            # Handle orchestrator pings in background
            async def handle_orchestrator_pings():
                try:
                    async for msg in orchestrator_ws:
                        data = json.loads(msg)
                        if data.get("type") == "ping":
                            await orchestrator_ws.send(json.dumps({"type": "pong"}))
                except Exception:
                    pass

            # Start ping handler
            ping_task = asyncio.create_task(handle_orchestrator_pings())

        async for message in websocket.iter_text():
            try:
                # HTMX ws-send serializes forms as JSON
                data = json.loads(message)

                # Extract session and user message
                session_id = data.get("session_id")
                user_message = data.get("message", "").strip()
                effort = data.get("effort", "medium")
                format_type = data.get("format", "text")

                if not user_message:
                    continue

                # Initialize session if needed
                if session_id not in sessions:
                    sessions[session_id] = []

                # Add user message to history
                sessions[session_id].append({"role": "user", "content": user_message})

                # Render user message HTML and update status
                user_html = templates.get_template("message_user.html").render(
                    content=format_content(user_message)
                )
                status_html = templates.get_template("status_update.html").render(
                    status_class="processing", status_text="Thinking..."
                )
                # Send both fragments together
                await websocket.send_text(user_html + status_html)

                # Mock mode: Generate response without hitting backend
                if MOCK_MODE:
                    # Simulate processing delay
                    delay = {"none": 0.5, "low": 1.0, "medium": 2.0, "high": 3.0}.get(effort, 2.0)
                    await asyncio.sleep(delay)

                    response_data = generate_mock_response(user_message, effort)
                else:
                    # Real mode: Send to orchestrator
                    # Build request for orchestrator
                    messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in sessions[session_id]
                    ]

                    # Encrypt messages
                    if fernet:
                        for msg in messages:
                            encrypted_token = fernet.encrypt(msg["content"].encode())
                            msg["content"] = encrypted_token.decode("utf-8")

                    request_data = {
                        "model": MODEL_NAME,
                        "messages": messages,
                        "thinking_effort": effort,
                    }

                    # Add format config
                    if format_type == "text":
                        request_data["extra_body"] = {
                            "structured_outputs": {
                                "json": {
                                    "type": "object",
                                    "properties": {"response": {"type": "string"}},
                                    "required": ["response"],
                                }
                            }
                        }
                    else:
                        request_data["response_format"] = {"type": "json_object"}

                    # Get timeout based on effort
                    timeouts = {"none": 30, "low": 60, "medium": 180, "high": 600}

                    request = {
                        "request": request_data,
                        "client_request_id": str(uuid4()),
                        "timeout_seconds": timeouts.get(effort, 180),
                    }

                    # Send to orchestrator
                    if not orchestrator_ws:
                        raise RuntimeError("Orchestrator WebSocket not connected")
                    await orchestrator_ws.send(json.dumps(request))

                    # Wait for response
                    response_text = await orchestrator_ws.recv()
                    response_data = json.loads(response_text)

                if response_data.get("status") == "completed":
                    result = response_data.get("result", {})
                    choice = result["choices"][0]

                    # Decrypt content
                    content = choice["message"]["content"]
                    thinking = choice.get("thinking")

                    if fernet:
                        if content:
                            content = fernet.decrypt(content.encode()).decode()
                        if thinking:
                            thinking = fernet.decrypt(thinking.encode()).decode()

                    # Unwrap JSON-wrapped text
                    if format_type == "text":
                        try:
                            parsed = json.loads(content)
                            if parsed.get("response"):
                                content = parsed["response"]
                        except (json.JSONDecodeError, KeyError):
                            pass

                    # Add to session history
                    sessions[session_id].append({"role": "assistant", "content": content})

                    # Render assistant message HTML and status update
                    thinking_id = f"thinking-{int(time.time() * 1000)}"
                    assistant_html = templates.get_template("message_assistant.html").render(
                        content=format_content(content),
                        thinking=thinking,
                        thinking_id=thinking_id,
                    )
                    status_html = templates.get_template("status_update.html").render(
                        status_class="connected", status_text="Connected"
                    )
                    # Send both fragments together
                    await websocket.send_text(assistant_html + status_html)

                elif response_data.get("status") in ["error", "timeout"]:
                    error_html = templates.get_template("message_error.html").render(
                        error=response_data.get("error", "Request failed")
                    )
                    status_html = templates.get_template("status_update.html").render(
                        status_class="connected", status_text="Connected"
                    )
                    await websocket.send_text(error_html + status_html)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Message handling error: {e}")
                error_html = templates.get_template("message_error.html").render(
                    error=f"Error: {str(e)}"
                )
                await websocket.send_text(error_html)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if ping_task:
            ping_task.cancel()
        if orchestrator_ws:
            await orchestrator_ws.close()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main chat interface"""
    # Generate new session ID
    session_id = str(uuid4())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_name": MODEL_NAME,
            "encryption_enabled": fernet is not None,
            "session_id": session_id,
        },
    )


@app.post("/clear")
async def clear_session(session_id: str = Form(...)):
    """Clear conversation history for a session"""
    if session_id in sessions:
        del sessions[session_id]
    return HTMLResponse("")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}
