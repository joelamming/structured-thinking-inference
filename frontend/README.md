# Frontend Chat Interface

**HTMX-based** FastAPI server that proxies WebSocket connections between browser and the orchestrator.

## Architecture

```
Browser <--HTMX WebSocket--> Frontend Server <--WebSocket+Fernet--> Orchestrator
         (HTML fragments)                      (encrypted JSON)
```

**Key Design:**
- **Server renders HTML** - All conversation state lives server-side in memory
- **HTMX handles UI** - WebSocket extension manages bidirectional communication
- **Minimal JS** - ~110 lines for keyboard shortcuts, thinking toggle, auto-scroll
- **Fernet encryption** - Transparent encryption/decryption between frontend and orchestrator

## Setup

```bash
pip install -r requirements.txt

# Configuration
export ORCH_ENCRYPTION_KEY="your-fernet-key"
export ORCHESTRATOR_WS_URL="ws://your-orchestrator:8005/ws/completions"
export ORCH_API_KEY="your-api-key"
export MODEL_NAME="your-model-name"

# Optional: Enable mock mode to avoid burning GPUs while prototyping
export MOCK_MODE="true"  # Set to "false" or unset for production

./run.sh
```

Open `http://localhost:8080`

## Mock Mode

**Perfect for prototyping without burning GPU credits!**

Set `MOCK_MODE=true` to enable a simulated backend that:
- ✅ Generates realistic mock responses instantly
- ✅ Simulates thinking process based on effort level
- ✅ Still uses encryption (if configured) to test the full flow
- ✅ No connection to orchestrator or vLLM needed
- ✅ All UI features work exactly the same

Mock responses clearly indicate they're simulated and show the original user message.

**Example:**
```bash
export MOCK_MODE="true"
./run.sh
# Frontend will display "Connected (Mock Mode)" in the status indicator
```

To switch to production:
```bash
export MOCK_MODE="false"
# or simply unset MOCK_MODE
./run.sh
```

## Files

### Backend
- `app.py` - FastAPI server with HTMX WebSocket handler (272 lines)
  - Session-based conversation management
  - HTML template rendering
  - Encryption/decryption proxy

### Frontend
- `templates/index.html` - Main page with HTMX WebSocket connection
- `templates/message_*.html` - HTML partials for chat messages (with `hx-swap-oob`)
- `templates/status_update.html` - Status indicator partial
- `static/script.js` - Minimal JS helpers (~110 lines)
- `static/style.css` - GitHub dark theme
- `static/htmx.min.js` - HTMX core library
- `static/htmx-ext-ws.js` - HTMX WebSocket extension

## How It Works

### 1. Form Submission (HTMX)
```html
<form ws-send>
    <input type="hidden" name="session_id" value="...">
    <textarea name="message"></textarea>
    <select name="effort"></select>
    <select name="format"></select>
</form>
```

HTMX automatically:
- Serializes form as JSON
- Sends via WebSocket
- No custom JS needed!

### 2. Server Processes Request
- Maintains conversation history per session
- Encrypts messages with Fernet
- Forwards to orchestrator
- Waits for response

### 3. Server Sends HTML Fragments
```html
<!-- Multiple fragments in one message -->
<div id="messages" hx-swap-oob="beforeend">
    <div class="message assistant">...</div>
</div>
<div id="status" hx-swap-oob="true">Connected</div>
```

HTMX automatically:
- Parses `hx-swap-oob` attributes
- Swaps content into correct elements
- No DOM manipulation in JS!

## What Changed from Pure JS

**Before (407 lines of JS):**
- Manual WebSocket connection management
- Manual JSON parsing/serialization
- Manual DOM manipulation
- localStorage for conversation persistence
- Complex state management

**After (110 lines of JS):**
- HTMX handles WebSocket connection
- HTMX handles form serialization
- HTMX handles DOM updates
- Server-side session storage
- JS only for: keyboard shortcuts, thinking toggle, status updates, clear button


# Linters and that

```bash
uv run ruff check

uv run mypy .

uv run ruff format --check

uv run ruff format
```
