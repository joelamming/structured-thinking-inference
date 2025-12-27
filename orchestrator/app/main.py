import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Optional, List, Tuple, TypedDict, cast
import os

import aiofiles  # type: ignore[import-untyped]
import httpx
from cryptography.fernet import Fernet
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from httpx import Timeout

from orchestrator.app.config import get_settings, Settings
from orchestrator.app.jobs import InMemoryJobStore
from orchestrator.app.metrics import MetricsSampler
from orchestrator.app.models import (
    EmbeddingRequest,
    ChatRequest,
    CompletionWebSocketRequest,
    OCRRequest,
    OCRWebSocketRequest,
)


settings: Settings = get_settings()
logger = logging.getLogger("custom_vllm.orchestrator")
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

THINKING_EFFORT_BUDGETS = {
    "none": 0,
    "low": 500,
    "medium": 2000,
    "high": 16000,
}
CUTOFF_NOTICE = "...\nTime's up; here's the answer."
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS") or "0")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
file_log_lock = asyncio.Lock()
http_client: Optional[httpx.AsyncClient] = None
job_store: Optional[InMemoryJobStore] = None
metrics_sampler: Optional[MetricsSampler] = None
worker_tasks: List[asyncio.Task] = []
metrics_task: Optional[asyncio.Task] = None
cleanup_task: Optional[asyncio.Task] = None
fernet = Fernet(
    settings.encryption_key.encode("utf-8")
    if isinstance(settings.encryption_key, str)
    else settings.encryption_key
)


async def save_log(record: Dict[str, Any]) -> None:
    if not settings.enable_request_logs:
        return
    async with file_log_lock:
        async with aiofiles.open(settings.log_file_path, "a") as log_file:
            await log_file.write(json.dumps(record) + "\n")


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    if not api_key or api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key"
        )
    return api_key


def get_templates_directory() -> str:
    from pathlib import Path

    static_dir = Path(__file__).resolve().parent.parent / "static"
    return str(static_dir)


templates = Jinja2Templates(directory=get_templates_directory())


async def create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=Timeout(360.0),
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
    )


def ws_error_payload(
    error: str,
    *,
    request_id: Optional[str] = None,
    error_code: str = "error",
    retry_after_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "error",
        "request_id": request_id,
        "error": error,
        "error_code": error_code,
    }
    if retry_after_seconds is not None:
        payload["retry_after_seconds"] = retry_after_seconds
    return payload


def ws_timeout_payload(request_id: str) -> Dict[str, Any]:
    return {"status": "timeout", "request_id": request_id, "error": "Request timed out"}


async def worker_loop(worker_id: int):
    """
    Worker task that processes jobs from the queue.
    Multiple workers can run concurrently to handle parallel requests.
    """
    assert job_store and http_client
    completions_url = settings.llm_server_url
    logger.debug(f"Worker {worker_id} started")
    while True:
        active_incremented = False
        try:
            job_tuple = await job_store.get_next_job(timeout_seconds=1)
            if not job_tuple:
                continue
            job_id, job_type, request_payload, client_request_id = job_tuple
            # Skip if canceled before processing
            status_record = await job_store.get_status(job_id)
            if status_record and status_record.get("status") == "cancelled":
                continue
            await job_store.increment_active()
            active_incremented = True
            await job_store.update_status(job_id, "running", started_ts=time.time())

            logger.debug(
                f"Worker {worker_id} processing job {job_id} (type: {job_type})"
            )

            if job_type == "ocr":
                ocr_request = OCRRequest.model_validate(request_payload)
                result_payload = await process_ocr_request(
                    ocr_request,
                    job_id=job_id,
                    client_request_id=client_request_id,
                )
            elif job_type == "chat":
                chat_request = ChatRequest.model_validate(request_payload)
                result_payload = await process_chat_request(
                    chat_request,
                    job_id=job_id,
                    client_request_id=client_request_id,
                    completions_url=completions_url,
                )
            else:
                raise ValueError(f"Unknown job_type '{job_type}'")

            await job_store.update_status(job_id, "completed", finished_ts=time.time())
            await job_store.publish_result(
                job_id,
                {"status": "completed", "request_id": job_id, "result": result_payload},
            )
            await save_log(
                {
                    "request_id": job_id,
                    "client_request_id": client_request_id,
                    "status": "completed",
                    "job_type": job_type,
                    "response_metadata": result_payload.get("usage"),
                    "timestamp": time.time(),
                }
            )
        except Exception as exc:
            logger.exception(f"Worker {worker_id} error")
            if "job_id" in locals():
                await job_store.update_status(job_id, "error", error=str(exc))
                await job_store.publish_result(
                    job_id,
                    {"status": "error", "request_id": job_id, "error": str(exc)},
                )
                await save_log(
                    {
                        "request_id": job_id,
                        "status": "error",
                        "error": str(exc),
                        "job_type": locals().get("job_type", "unknown"),
                        "timestamp": time.time(),
                    }
                )
        finally:
            if active_incremented:
                await job_store.decrement_active()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        http_client, \
        job_store, \
        metrics_sampler, \
        worker_tasks, \
        metrics_task, \
        cleanup_task

    # Determine number of concurrent workers
    # Default to 20 to handle typical concurrent load, configurable via env var
    num_workers = int(os.getenv("NUM_WORKERS", "20"))
    logger.info(f"Starting {num_workers} concurrent worker tasks")

    http_client = await create_http_client()
    job_store = InMemoryJobStore(settings)
    metrics_sampler = MetricsSampler(settings, job_store, http_client)

    # Spawn multiple concurrent workers
    worker_tasks.clear()
    for worker_id in range(num_workers):
        task = asyncio.create_task(worker_loop(worker_id))
        worker_tasks.append(task)

    metrics_task = asyncio.create_task(metrics_sampler.run())
    cleanup_task = asyncio.create_task(cleanup_loop())

    try:
        yield
    finally:
        # Cancel all worker tasks
        for task in worker_tasks:
            if task:
                task.cancel()
        if metrics_task:
            metrics_task.cancel()
        if cleanup_task:
            cleanup_task.cancel()

        # Wait for cancellation to complete
        for task in worker_tasks:
            with suppress(asyncio.CancelledError):
                await task
        if metrics_task:
            with suppress(asyncio.CancelledError):
                await metrics_task
        if cleanup_task:
            with suppress(asyncio.CancelledError):
                await cleanup_task

        if http_client:
            await http_client.aclose()


app = FastAPI(
    title="Custom vLLM Orchestrator",
    version="0.1.0",
    lifespan=lifespan,
    openapi_version=settings.openapi_version,
)

app.mount("/static", StaticFiles(directory=get_templates_directory()), name="static")


def format_prompt_for_llm(messages: List[Dict[str, str]], model_name: str) -> str:
    model_short_name = model_name.lower()
    full_prompt: List[str] = []
    if "qwen3" in model_short_name and "deepseek" not in model_short_name:
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            full_prompt.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        full_prompt.append("<|im_start|>assistant\n<think>\n")
        return "\n".join(full_prompt)
    if "deepseek" in model_short_name:
        user_prompt = ""
        if messages and messages[-1].get("role") == "user":
            user_prompt = messages[-1]["content"]
        return "<|begin of sentence|> " + user_prompt + " <|Assistant|><think>\n"
    raise ValueError(f"Unknown model type '{model_name}' for custom prompt formatting.")


def derive_thinking_budget(effort: str) -> int:
    return THINKING_EFFORT_BUDGETS.get(effort, THINKING_EFFORT_BUDGETS["medium"])


async def run_thinking_prompt(
    prompt: str,
    request: ChatRequest,
    job_id: str,
    client_request_id: Optional[str],
    request_suffix: str,
    max_tokens: int,
) -> Tuple[str, str]:
    """
    Run a single thinking completion with a dedicated budget.

    Returns the thinking text and the finish_reason.
    """
    assert http_client is not None
    thinking_request = {
        "model": request.model,
        "prompt": prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": max_tokens,
        "stream": False,
        "stop": ["</think>"],
        "repetition_penalty": request.repetition_penalty,
    }
    thinking_response = await http_client.post(
        settings.llm_server_url,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": f"{job_id}-{request_suffix}",
            "X-Client-Request-ID": client_request_id or "none",
        },
        json=thinking_request,
        timeout=None,
    )
    thinking_response.raise_for_status()
    thinking_data = thinking_response.json()
    thinking_text = ""
    finish_reason = "stop"
    if "choices" in thinking_data and thinking_data["choices"]:
        choice = thinking_data["choices"][0]
        thinking_text = choice.get("text", "") or ""
        finish_reason = choice.get("finish_reason", "stop")
    if finish_reason == "length" and "</think>" not in thinking_text:
        trimmed = thinking_text.rstrip("\n")
        prefix = f"{trimmed}\n" if trimmed else ""
        thinking_text = f"{prefix}{CUTOFF_NOTICE}\n"
    return thinking_text, finish_reason


async def process_chat_request(
    request: ChatRequest,
    job_id: str,
    client_request_id: Optional[str],
    completions_url: str,
) -> Dict[str, Any]:
    assert http_client is not None
    request_fernet = fernet
    logger_prefix = f"[Job {job_id}"
    if client_request_id:
        logger_prefix += f", Client {client_request_id[:8]}"
    logger_prefix += "] "

    decrypted_messages: List[Dict[str, str]] = []
    for message in request.messages:
        encrypted_content_token = message.get("content", "")
        if isinstance(encrypted_content_token, str):
            # Fernet token is already base64-encoded, just needs to be bytes
            decrypted_content = request_fernet.decrypt(
                encrypted_content_token.encode()
            ).decode("utf-8")
            decrypted_messages.append(
                {"role": message.get("role", "user"), "content": decrypted_content}
            )

    formatted_prompt_for_tokenisation = format_prompt_for_llm(
        decrypted_messages, request.model
    )
    needs_two_step = False
    structured_outputs: Optional[Dict[str, Any]] = None

    if request.extra_body:
        extra = request.extra_body
        if "structured_outputs" in extra:
            structured_outputs = cast(Dict[str, Any], extra["structured_outputs"])
            needs_two_step = True

    if not structured_outputs and request.response_format:
        rf = request.response_format
        if isinstance(rf, dict):
            if rf.get("type") == "json_object":
                structured_outputs = {"json": {"type": "object"}}
                needs_two_step = True
            elif rf.get("type") == "json_schema":
                json_schema_obj = rf.get("json_schema", {}).get("schema") or rf.get(
                    "schema"
                )
                if json_schema_obj:
                    structured_outputs = {"json": json_schema_obj}
                    needs_two_step = True

    if not needs_two_step:
        raise ValueError("No guided decoding requested")

    effort = getattr(request, "thinking_effort", "medium") or "medium"
    thinking_budget = derive_thinking_budget(effort)
    final_max_tokens = FINAL_MAX_TOKENS
    formatted_prompt = formatted_prompt_for_tokenisation
    base_prompt_for_later_steps = formatted_prompt.rsplit("<think>", 1)[0]

    completions_single_url = completions_url

    thinking_content_1 = ""
    if thinking_budget > 0:
        thinking_content_1, _ = await run_thinking_prompt(
            formatted_prompt,
            request,
            job_id,
            client_request_id,
            "thinking-1",
            thinking_budget,
        )

    prompt_step_2 = formatted_prompt + thinking_content_1 + "</think>\n"
    guided_request_1: Dict[str, Any] = {
        "model": request.model,
        "prompt": prompt_step_2,
        "temperature": 0.0,
        "top_p": request.top_p,
        "max_tokens": final_max_tokens,
        "stream": False,
        "repetition_penalty": request.repetition_penalty,
    }
    if request.stop:
        guided_request_1["stop"] = request.stop
    if structured_outputs:
        guided_request_1["structured_outputs"] = structured_outputs
        if "regex" in structured_outputs:
            if "stop" not in guided_request_1 or not guided_request_1["stop"]:
                guided_request_1["stop"] = ["</s>"]
            elif "</s>" not in guided_request_1["stop"]:
                guided_request_1["stop"].append("</s>")

    response_1 = await http_client.post(
        completions_single_url,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": f"{job_id}-guided-1",
            "X-Client-Request-ID": client_request_id or "none",
        },
        json=guided_request_1,
        timeout=None,
    )
    if response_1.status_code != 200:
        logger.error(f"vLLM error response: {response_1.text}")
    response_1.raise_for_status()
    resp1_json = response_1.json()
    structured_content_1 = ""
    if isinstance(resp1_json, dict) and resp1_json.get("choices"):
        first_choice = resp1_json["choices"][0]
        structured_content_1 = first_choice.get("text") or first_choice.get(
            "message", {}
        ).get("content", "")
    if not structured_content_1:
        raise ValueError("Guided decoding (step 2) returned no content")

    if effort == "none":
        isolated_response_data = json.loads(json.dumps(resp1_json))
        if isolated_response_data.get("choices"):
            choice = isolated_response_data["choices"][0]
            text_content = choice.get("text") or choice.get("message", {}).get(
                "content", ""
            )
            choice["message"] = {
                "role": "assistant",
                "content": text_content or structured_content_1,
                "thinking": "",
            }
            choice["finish_reason"] = choice.get("finish_reason", "stop")
            choice.pop("text", None)

        if isolated_response_data.get("choices"):
            message = isolated_response_data["choices"][0]["message"]
            content_bytes = message["content"].encode("utf-8")
            encrypted_content = request_fernet.encrypt(content_bytes)
            message["content"] = encrypted_content.decode(
                "utf-8"
            )  # Fernet token is already base64
            if "thinking" in message and message["thinking"]:
                thinking_bytes = message["thinking"].encode("utf-8")
                encrypted_thinking = request_fernet.encrypt(thinking_bytes)
                message["thinking"] = encrypted_thinking.decode(
                    "utf-8"
                )  # Fernet token is already base64
        return isolated_response_data

    correction_thinking_body = (
        f"{thinking_content_1.strip()}\n\nMy final output will therefore be:\n\n{structured_content_1.strip()}"
        f"\n\nWait, have I made a mistake here?\n"
    )
    prompt_step_3 = (
        base_prompt_for_later_steps + f"<think>\n{correction_thinking_body}\n"
    )

    thinking_correction = ""
    thinking_correction, _ = await run_thinking_prompt(
        prompt_step_3,
        request,
        job_id,
        client_request_id,
        "thinking-2",
        thinking_budget,
    )

    thinking_content = f"{correction_thinking_body}\n{thinking_correction}"
    final_prompt = (
        base_prompt_for_later_steps + f"<think>\n{thinking_content}</think>\n"
    )

    guided_request: Dict[str, Any] = {
        "model": request.model,
        "prompt": final_prompt,
        "temperature": 0.0,
        "top_p": request.top_p,
        "max_tokens": final_max_tokens,
        "stream": False,
        "repetition_penalty": request.repetition_penalty,
    }
    if request.stop:
        guided_request["stop"] = request.stop
    if structured_outputs:
        guided_request["structured_outputs"] = structured_outputs
        if "regex" in structured_outputs:
            if "stop" not in guided_request or not guided_request["stop"]:
                guided_request["stop"] = ["</s>"]
            elif "</s>" not in guided_request["stop"]:
                guided_request["stop"].append("</s>")

    response = await http_client.post(
        completions_single_url,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": job_id,
            "X-Client-Request-ID": client_request_id or "none",
        },
        json=guided_request,
        timeout=None,
    )
    if response.status_code != 200:
        logger.error(f"vLLM error response: {response.text}")
    response.raise_for_status()
    response_data = response.json()
    isolated_response_data = json.loads(json.dumps(response_data))
    this_request_content = None
    if isolated_response_data.get("choices"):
        choice = isolated_response_data["choices"][0]
        text_content = choice.get("text")
        if text_content is None and choice.get("message"):
            text_content = choice["message"].get("content")
        this_request_content = text_content
        choice["message"] = {
            "role": "assistant",
            "content": this_request_content,
            "thinking": thinking_content,
        }
        choice["finish_reason"] = choice.get("finish_reason", "stop")
        choice.pop("text", None)

    if isolated_response_data.get("choices"):
        message = isolated_response_data["choices"][0]["message"]
        content_bytes = message["content"].encode("utf-8")
        encrypted_content = request_fernet.encrypt(content_bytes)
        message["content"] = encrypted_content.decode(
            "ascii"
        )  # Fernet token is already base64
        if "thinking" in message and message["thinking"]:
            thinking_bytes = message["thinking"].encode("utf-8")
            encrypted_thinking = request_fernet.encrypt(thinking_bytes)
            message["thinking"] = encrypted_thinking.decode(
                "ascii"
            )  # Fernet token is already base64

    return isolated_response_data


async def process_ocr_request(
    request: OCRRequest,
    job_id: str,
    client_request_id: Optional[str],
) -> Dict[str, Any]:
    """
    Process an OCR request by decrypting messages, calling vLLM DeepSeek-OCR,
    and encrypting the response.

    Unlike process_chat_request, this is a simple pass-through:
    - No multi-step thinking pipeline
    - Direct call to vLLM with image data
    - Returns encrypted markdown/HTML output
    """
    assert http_client is not None
    request_fernet = fernet
    logger_prefix = f"[OCR Job {job_id}"
    if client_request_id:
        logger_prefix += f", Client {client_request_id[:8]}"
    logger_prefix += "] "

    # Decrypt message contents (including base64 image data)
    decrypted_messages: List[Dict[str, Any]] = []
    for message in request.messages:
        role = message.get("role", "user")
        content = message.get("content", [])

        # Handle both string content and array content (for images)
        if isinstance(content, str):
            # Simple text content - decrypt it
            decrypted_content = request_fernet.decrypt(content.encode()).decode("utf-8")
            decrypted_messages.append({"role": role, "content": decrypted_content})
        elif isinstance(content, list):
            # Array content with potentially image_url and text parts
            decrypted_content_parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "text":
                        # Decrypt text content
                        encrypted_text = part.get("text", "")
                        decrypted_text = request_fernet.decrypt(
                            encrypted_text.encode()
                        ).decode("utf-8")
                        decrypted_content_parts.append(
                            {"type": "text", "text": decrypted_text}
                        )
                    elif part_type == "image_url":
                        # Decrypt the image URL (which contains base64 data)
                        image_url_obj = part.get("image_url", {})
                        encrypted_url = image_url_obj.get("url", "")
                        decrypted_url = request_fernet.decrypt(
                            encrypted_url.encode()
                        ).decode("utf-8")
                        image_url_part: Dict[str, Any] = {
                            "type": "image_url",
                            "image_url": {"url": decrypted_url},
                        }
                        decrypted_content_parts.append(image_url_part)
                    else:
                        # Unknown type, pass through
                        decrypted_content_parts.append(part)
            decrypted_messages.append(
                {"role": role, "content": decrypted_content_parts}
            )
        else:
            # Unknown format, pass through
            decrypted_messages.append({"role": role, "content": content})

    # Prepare vLLM request (OpenAI chat/completions format)
    ocr_url = settings.ocr_server_url or settings.llm_server_url.replace(
        "/v1/completions", "/v1/chat/completions"
    )

    vllm_request: Dict[str, Any] = {
        "model": request.model,
        "messages": decrypted_messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }

    # Add extra_body if provided (for vllm_xargs like ngram_size, etc.)
    if request.extra_body:
        for key, value in request.extra_body.items():
            vllm_request[key] = value

    logger.debug(f"{logger_prefix}Sending OCR request to {ocr_url}")

    # Call vLLM
    response = await http_client.post(
        ocr_url,
        headers={
            "Content-Type": "application/json",
            "X-Request-ID": job_id,
            "X-Client-Request-ID": client_request_id or "none",
        },
        json=vllm_request,
        timeout=None,
    )

    if response.status_code != 200:
        logger.error(f"{logger_prefix}vLLM error response: {response.text}")
    response.raise_for_status()

    response_data = response.json()

    # Encrypt the response content
    if response_data.get("choices"):
        for choice in response_data["choices"]:
            message = choice.get("message", {})
            content = message.get("content", "")

            if content:
                content_bytes = content.encode("utf-8")
                encrypted_content = request_fernet.encrypt(content_bytes)
                message["content"] = encrypted_content.decode("ascii")

    return response_data


@app.websocket("/ws/completions")
async def websocket_completions(websocket: WebSocket):
    """
    WebSocket endpoint for completion requests.

    Architecture: Uses concurrent tasks for robust keepalive handling:
    - receiver_task: Receives all client messages and routes them appropriately
    - keepalive_task: Sends periodic pings and monitors client activity
    - Main loop: Processes job results and sends them to the client

    This avoids blocking receives that compete for messages and ensures
    keepalive pings don't interfere with request/response handling.
    """
    api_key = websocket.headers.get("X-API-Key")
    if api_key != settings.api_key:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    assert job_store is not None
    await job_store.increment_ws_connections()

    # Shared state for concurrent tasks
    pending_jobs: dict[str, float] = {}  # job_id -> deadline timestamp
    incoming_requests: asyncio.Queue[dict] = asyncio.Queue(
        maxsize=settings.ws_incoming_queue_maxsize
    )
    client_request_ids: dict[str, str] = {}  # client_request_id -> job_id
    connection_alive = True
    last_client_activity = time.time()
    shutdown_event = asyncio.Event()

    # Keepalive configuration
    ping_interval_seconds = 15
    # Allow some missed pings but also check activity - if we've received any
    # message recently, the connection is healthy even without explicit pongs
    activity_timeout_seconds = ping_interval_seconds * 3  # 45s of inactivity = dead

    async def receiver_task():
        """
        Receive all messages from the client and route them.
        Runs until connection closes or shutdown is signaled.
        """
        nonlocal connection_alive, last_client_activity
        try:
            while connection_alive and not shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0,  # Short timeout to check shutdown_event
                    )
                    last_client_activity = time.time()
                    message_size = len(message.encode("utf-8"))
                    if message_size > settings.max_ws_message_bytes:
                        logger.warning(
                            "WebSocket message too large (%d bytes > %d bytes)",
                            message_size,
                            settings.max_ws_message_bytes,
                        )
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    "Message too large",
                                    error_code="message_too_large",
                                )
                            )
                        except Exception:
                            pass
                        try:
                            await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                        except Exception:
                            pass
                        connection_alive = False
                        break
                    try:
                        payload = json.loads(message)
                        if not isinstance(payload, dict):
                            raise ValueError("Payload must be a JSON object")
                        msg_type = payload.get("type")

                        if msg_type == "pong":
                            # Pong received - activity already updated above
                            continue
                        elif msg_type == "cancel":
                            # Client wants to cancel a job
                            cancel_request_id = payload.get("request_id")
                            cancel_client_request_id = payload.get("client_request_id")
                            resolved_job_id: Optional[str] = None
                            for candidate in (
                                cancel_request_id,
                                cancel_client_request_id,
                            ):
                                if not candidate:
                                    continue
                                if candidate in pending_jobs:
                                    resolved_job_id = candidate
                                    break
                                if candidate in client_request_ids:
                                    resolved_job_id = client_request_ids.get(candidate)
                                    break
                            if resolved_job_id:
                                await job_store.cancel_job(resolved_job_id)
                                pending_jobs.pop(resolved_job_id, None)
                                for k, v in list(client_request_ids.items()):
                                    if v == resolved_job_id:
                                        client_request_ids.pop(k, None)
                            continue
                        else:
                            # Assume it's a completion request
                            try:
                                incoming_requests.put_nowait(payload)
                            except asyncio.QueueFull:
                                try:
                                    await websocket.send_json(
                                        ws_error_payload(
                                            "Too many pending requests",
                                            error_code="backpressure",
                                            retry_after_seconds=1,
                                        )
                                    )
                                except Exception:
                                    pass
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid JSON from client: %s", exc)
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    f"Invalid JSON: {exc}", error_code="invalid_json"
                                )
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.warning("Invalid message from client: %s", exc)
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    f"Invalid message: {exc}",
                                    error_code="invalid_message",
                                )
                            )
                        except Exception:
                            pass
                except asyncio.TimeoutError:
                    # No message received, just loop to check shutdown_event
                    continue
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected in receiver")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Receiver task error: %s", exc)
        finally:
            connection_alive = False

    async def keepalive_task():
        """
        Send periodic pings and monitor client activity.
        Closes connection if client appears dead.
        """
        nonlocal connection_alive
        try:
            while connection_alive and not shutdown_event.is_set():
                await asyncio.sleep(ping_interval_seconds)
                if shutdown_event.is_set() or not connection_alive:
                    break

                # Check if we've had any activity recently
                time_since_activity = time.time() - last_client_activity
                if time_since_activity > activity_timeout_seconds:
                    logger.warning(
                        "No client activity for %.1fs, closing connection",
                        time_since_activity,
                    )
                    connection_alive = False
                    break

                # Send application-level ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception as exc:
                    logger.debug("Failed to send ping: %s", exc)
                    connection_alive = False
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Keepalive task error: %s", exc)
            connection_alive = False

    async def result_waiter_task(job_id: str, timeout_seconds: int):
        """Wait for a job result and return it, or None if timed out/cancelled."""
        try:
            result = await job_store.wait_for_result(job_id, timeout_seconds)
            return result
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None

    # Start background tasks
    receiver = asyncio.create_task(receiver_task())
    keepalive = asyncio.create_task(keepalive_task())
    active_waiters: dict[str, asyncio.Task] = {}  # job_id -> waiter task

    try:
        while connection_alive:
            # Process incoming requests (drain queue with backpressure checks)
            drained = 0
            while drained < 50:
                try:
                    payload = incoming_requests.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained += 1
                try:
                    if len(pending_jobs) >= settings.max_pending_jobs_per_connection:
                        await websocket.send_json(
                            ws_error_payload(
                                "Too many in-flight requests for this connection",
                                error_code="too_many_requests",
                                retry_after_seconds=1,
                            )
                        )
                        continue

                    counters = await job_store.get_counters()
                    if counters["queue_depth"] >= settings.max_queue_depth:
                        await websocket.send_json(
                            ws_error_payload(
                                "Server overloaded (queue full)",
                                error_code="server_overloaded",
                                retry_after_seconds=1,
                            )
                        )
                        continue

                    ws_request = CompletionWebSocketRequest.model_validate(payload)
                    timeout_seconds = int(
                        ws_request.timeout_seconds or settings.job_timeout_seconds
                    )
                    job_id = await job_store.enqueue_job(
                        "chat",
                        ws_request.request.model_dump(),
                        ws_request.client_request_id,
                        timeout_seconds=timeout_seconds,
                    )
                    if ws_request.client_request_id:
                        client_request_ids[ws_request.client_request_id] = job_id
                    pending_jobs[job_id] = time.time() + timeout_seconds
                    # Start a waiter task for this job
                    active_waiters[job_id] = asyncio.create_task(
                        result_waiter_task(job_id, timeout_seconds)
                    )
                except Exception as exc:
                    logger.warning("Invalid request payload: %s", exc)
                    try:
                        await websocket.send_json(
                            ws_error_payload(
                                f"Invalid payload: {exc}",
                                error_code="invalid_payload",
                            )
                        )
                    except Exception:
                        pass

            # Check for completed job waiters
            completed_waiters = [
                jid for jid, task in active_waiters.items() if task.done()
            ]
            for job_id in completed_waiters:
                task = active_waiters.pop(job_id)
                pending_jobs.pop(job_id, None)
                for k, v in list(client_request_ids.items()):
                    if v == job_id:
                        client_request_ids.pop(k, None)
                try:
                    result = task.result()
                    if result is not None:
                        await websocket.send_json(result)
                    else:
                        # Timed out
                        await job_store.timeout_job(job_id)
                        await websocket.send_json(ws_timeout_payload(job_id))
                except asyncio.CancelledError:
                    await job_store.cancel_job(job_id)
                except Exception as exc:
                    logger.warning("Error processing job result: %s", exc)

            # Check for jobs that have exceeded their deadline
            now = time.time()
            expired_jobs = [
                jid for jid, deadline in pending_jobs.items() if now > deadline
            ]
            for job_id in expired_jobs:
                pending_jobs.pop(job_id, None)
                if job_id in active_waiters:
                    active_waiters[job_id].cancel()
                    active_waiters.pop(job_id, None)
                for k, v in list(client_request_ids.items()):
                    if v == job_id:
                        client_request_ids.pop(k, None)
                await job_store.timeout_job(job_id)
                try:
                    await websocket.send_json(ws_timeout_payload(job_id))
                except Exception:
                    pass

            # Brief sleep to prevent busy-loop
            await asyncio.sleep(0.1)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.exception("WebSocket handler error: %s", exc)
    finally:
        # Signal shutdown and clean up
        shutdown_event.set()
        connection_alive = False

        # Cancel background tasks
        receiver.cancel()
        keepalive.cancel()
        for task in active_waiters.values():
            task.cancel()

        with suppress(asyncio.CancelledError):
            await receiver
        with suppress(asyncio.CancelledError):
            await keepalive
        for task in active_waiters.values():
            with suppress(asyncio.CancelledError):
                await task

        # Cancel any outstanding jobs
        for job_id in list(pending_jobs.keys()):
            await job_store.cancel_job(job_id)

        await job_store.decrement_ws_connections()


@app.websocket("/ws/ocr")
async def websocket_ocr(websocket: WebSocket):
    """
    WebSocket endpoint for OCR requests using DeepSeek-OCR model.

    Handles image-to-markdown/HTML conversion requests with:
    - Symmetric encryption for image data and responses
    - Job queue for managing concurrent requests
    - Keepalive and cancellation support

    Uses the same architecture as /ws/completions but simpler processing
    (no multi-step thinking pipeline).
    """
    api_key = websocket.headers.get("X-API-Key")
    if api_key != settings.api_key:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    assert job_store is not None
    await job_store.increment_ws_connections()

    # Shared state for concurrent tasks
    pending_jobs: dict[str, float] = {}  # job_id -> deadline timestamp
    incoming_requests: asyncio.Queue[dict] = asyncio.Queue(
        maxsize=settings.ws_incoming_queue_maxsize
    )
    client_request_ids: dict[str, str] = {}  # client_request_id -> job_id
    connection_alive = True
    last_client_activity = time.time()
    shutdown_event = asyncio.Event()

    # Keepalive configuration
    ping_interval_seconds = 15
    activity_timeout_seconds = ping_interval_seconds * 3  # 45s of inactivity = dead

    async def receiver_task():
        """
        Receive all messages from the client and route them.
        Runs until connection closes or shutdown is signalled.
        """
        nonlocal connection_alive, last_client_activity
        try:
            while connection_alive and not shutdown_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=1.0,  # Short timeout to check shutdown_event
                    )
                    last_client_activity = time.time()
                    message_size = len(message.encode("utf-8"))
                    if message_size > settings.max_ws_message_bytes:
                        logger.warning(
                            "WebSocket message too large (%d bytes > %d bytes)",
                            message_size,
                            settings.max_ws_message_bytes,
                        )
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    "Message too large",
                                    error_code="message_too_large",
                                )
                            )
                        except Exception:
                            pass
                        try:
                            await websocket.close(code=status.WS_1009_MESSAGE_TOO_BIG)
                        except Exception:
                            pass
                        connection_alive = False
                        break
                    try:
                        payload = json.loads(message)
                        if not isinstance(payload, dict):
                            raise ValueError("Payload must be a JSON object")
                        msg_type = payload.get("type")

                        if msg_type == "pong":
                            # Pong received - activity already updated above
                            continue
                        elif msg_type == "cancel":
                            # Client wants to cancel a job
                            cancel_request_id = payload.get("request_id")
                            cancel_client_request_id = payload.get("client_request_id")
                            resolved_job_id: Optional[str] = None
                            for candidate in (
                                cancel_request_id,
                                cancel_client_request_id,
                            ):
                                if not candidate:
                                    continue
                                if candidate in pending_jobs:
                                    resolved_job_id = candidate
                                    break
                                if candidate in client_request_ids:
                                    resolved_job_id = client_request_ids.get(candidate)
                                    break
                            if resolved_job_id:
                                await job_store.cancel_job(resolved_job_id)
                                pending_jobs.pop(resolved_job_id, None)
                                for k, v in list(client_request_ids.items()):
                                    if v == resolved_job_id:
                                        client_request_ids.pop(k, None)
                            continue
                        else:
                            # Assume it's an OCR request
                            try:
                                incoming_requests.put_nowait(payload)
                            except asyncio.QueueFull:
                                try:
                                    await websocket.send_json(
                                        ws_error_payload(
                                            "Too many pending requests",
                                            error_code="backpressure",
                                            retry_after_seconds=1,
                                        )
                                    )
                                except Exception:
                                    pass
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid JSON from client: %s", exc)
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    f"Invalid JSON: {exc}", error_code="invalid_json"
                                )
                            )
                        except Exception:
                            pass
                    except Exception as exc:
                        logger.warning("Invalid message from client: %s", exc)
                        try:
                            await websocket.send_json(
                                ws_error_payload(
                                    f"Invalid message: {exc}",
                                    error_code="invalid_message",
                                )
                            )
                        except Exception:
                            pass
                except asyncio.TimeoutError:
                    # No message received, just loop to check shutdown_event
                    continue
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected in receiver")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Receiver task error: %s", exc)
        finally:
            connection_alive = False

    async def keepalive_task():
        """
        Send periodic pings and monitor client activity.
        Closes connection if client appears dead.
        """
        nonlocal connection_alive
        try:
            while connection_alive and not shutdown_event.is_set():
                await asyncio.sleep(ping_interval_seconds)
                if shutdown_event.is_set() or not connection_alive:
                    break

                # Check if we've had any activity recently
                time_since_activity = time.time() - last_client_activity
                if time_since_activity > activity_timeout_seconds:
                    logger.warning(
                        "No client activity for %.1fs, closing connection",
                        time_since_activity,
                    )
                    connection_alive = False
                    break

                # Send application-level ping
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception as exc:
                    logger.debug("Failed to send ping: %s", exc)
                    connection_alive = False
                    break
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning("Keepalive task error: %s", exc)
            connection_alive = False

    async def result_waiter_task(job_id: str, timeout_seconds: int):
        """Wait for a job result and return it, or None if timed out/cancelled."""
        try:
            result = await job_store.wait_for_result(job_id, timeout_seconds)
            return result
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            return None

    # Start background tasks
    receiver = asyncio.create_task(receiver_task())
    keepalive = asyncio.create_task(keepalive_task())
    active_waiters: dict[str, asyncio.Task] = {}  # job_id -> waiter task

    try:
        while connection_alive:
            # Process incoming requests (drain queue with backpressure checks)
            drained = 0
            while drained < 50:
                try:
                    payload = incoming_requests.get_nowait()
                except asyncio.QueueEmpty:
                    break
                drained += 1
                try:
                    if len(pending_jobs) >= settings.max_pending_jobs_per_connection:
                        await websocket.send_json(
                            ws_error_payload(
                                "Too many in-flight requests for this connection",
                                error_code="too_many_requests",
                                retry_after_seconds=1,
                            )
                        )
                        continue

                    counters = await job_store.get_counters()
                    if counters["queue_depth"] >= settings.max_queue_depth:
                        await websocket.send_json(
                            ws_error_payload(
                                "Server overloaded (queue full)",
                                error_code="server_overloaded",
                                retry_after_seconds=1,
                            )
                        )
                        continue

                    ws_request = OCRWebSocketRequest.model_validate(payload)
                    timeout_seconds = int(
                        ws_request.timeout_seconds or settings.job_timeout_seconds
                    )
                    job_id = await job_store.enqueue_job(
                        "ocr",
                        ws_request.request.model_dump(),
                        ws_request.client_request_id,
                        timeout_seconds=timeout_seconds,
                    )
                    if ws_request.client_request_id:
                        client_request_ids[ws_request.client_request_id] = job_id
                    pending_jobs[job_id] = time.time() + timeout_seconds
                    # Start a waiter task for this job
                    active_waiters[job_id] = asyncio.create_task(
                        result_waiter_task(job_id, timeout_seconds)
                    )
                except Exception as exc:
                    logger.warning("Invalid OCR request payload: %s", exc)
                    try:
                        await websocket.send_json(
                            ws_error_payload(
                                f"Invalid payload: {exc}",
                                error_code="invalid_payload",
                            )
                        )
                    except Exception:
                        pass

            # Check for completed job waiters
            completed_waiters = [
                jid for jid, task in active_waiters.items() if task.done()
            ]
            for job_id in completed_waiters:
                task = active_waiters.pop(job_id)
                pending_jobs.pop(job_id, None)
                for k, v in list(client_request_ids.items()):
                    if v == job_id:
                        client_request_ids.pop(k, None)
                try:
                    result = task.result()
                    if result is not None:
                        await websocket.send_json(result)
                    else:
                        # Timed out
                        await job_store.timeout_job(job_id)
                        await websocket.send_json(ws_timeout_payload(job_id))
                except asyncio.CancelledError:
                    await job_store.cancel_job(job_id)
                except Exception as exc:
                    logger.warning("Error processing OCR job result: %s", exc)

            # Check for jobs that have exceeded their deadline
            now = time.time()
            expired_jobs = [
                jid for jid, deadline in pending_jobs.items() if now > deadline
            ]
            for job_id in expired_jobs:
                pending_jobs.pop(job_id, None)
                if job_id in active_waiters:
                    active_waiters[job_id].cancel()
                    active_waiters.pop(job_id, None)
                for k, v in list(client_request_ids.items()):
                    if v == job_id:
                        client_request_ids.pop(k, None)
                await job_store.timeout_job(job_id)
                try:
                    await websocket.send_json(ws_timeout_payload(job_id))
                except Exception:
                    pass

            # Brief sleep to prevent busy-loop
            await asyncio.sleep(0.1)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as exc:
        logger.exception("OCR WebSocket handler error: %s", exc)
    finally:
        # Signal shutdown and clean up
        shutdown_event.set()
        connection_alive = False

        # Cancel background tasks
        receiver.cancel()
        keepalive.cancel()
        for task in active_waiters.values():
            task.cancel()

        with suppress(asyncio.CancelledError):
            await receiver
        with suppress(asyncio.CancelledError):
            await keepalive
        for task in active_waiters.values():
            with suppress(asyncio.CancelledError):
                await task

        # Cancel any outstanding jobs
        for job_id in list(pending_jobs.keys()):
            await job_store.cancel_job(job_id)

        await job_store.decrement_ws_connections()


@app.post("/embeddings")
async def embeddings_endpoint(
    request: EmbeddingRequest,
    api_key: str = Depends(verify_api_key),
):
    assert http_client is not None
    decrypted_texts: List[str] = []
    for text in request.input_texts:
        encrypted_bytes = base64.b64decode(text)
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        decrypted_texts.append(decrypted_bytes.decode("utf-8"))
    payload = {
        "model": request.model,
        "input": decrypted_texts,
        "encoding_format": request.encoding_format,
    }
    response = await http_client.post(
        settings.embedding_server_url,
        headers={"Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    return response.json()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


class ChartData(TypedDict):
    labels: List[str]
    avg_gen_throughput: List[float]
    gpu_cache_usage: List[float]
    memory_usage_percent: List[float]
    gpu_utilisation: List[float]


@app.get("/dashboard/metrics", response_class=HTMLResponse)
async def dashboard_metrics(request: Request):
    metrics_data: Dict[str, Any] = {"timestamp": "n/a"}
    empty_chart: ChartData = {
        "labels": [],
        "avg_gen_throughput": [],
        "gpu_cache_usage": [],
        "memory_usage_percent": [],
        "gpu_utilisation": [],
    }
    chart_data: ChartData = empty_chart
    if metrics_sampler and metrics_sampler.latest_metrics:
        metrics_data = metrics_sampler.latest_metrics.get("current", metrics_data)
        chart_data = cast(
            ChartData, metrics_sampler.latest_metrics.get("chart", empty_chart)
        )
    context = {
        "request": request,
        "metrics": metrics_data,
        "chart_data": chart_data,
    }
    return templates.TemplateResponse("metrics_partial.html", context)


async def cleanup_loop() -> None:
    """Periodically evict expired jobs/results from memory."""
    assert job_store is not None
    interval = max(60, settings.metrics_interval_seconds)  # at least once a minute
    max_age = settings.job_timeout_seconds
    while True:
        try:
            await job_store.cleanup_expired(max_age_seconds=max_age)
        except Exception as exc:
            logging.getLogger("custom_vllm.cleanup").warning("Cleanup error: %s", exc)
        await asyncio.sleep(interval)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
