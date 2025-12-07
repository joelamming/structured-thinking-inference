import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, Optional, List, Tuple
import os

import aiofiles
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

from .config import get_settings, Settings
from .jobs import InMemoryJobStore
from .metrics import MetricsSampler
from .models import EmbeddingRequest, ChatRequest, CompletionWebSocketRequest


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
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS"))

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
file_log_lock = asyncio.Lock()
http_client: Optional[httpx.AsyncClient] = None
job_store: Optional[InMemoryJobStore] = None
metrics_sampler: Optional[MetricsSampler] = None
worker_task: Optional[asyncio.Task] = None
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


async def worker_loop():
    assert job_store and http_client
    completions_url = settings.llm_server_url
    while True:
        active_incremented = False
        try:
            job_tuple = await job_store.get_next_job(timeout_seconds=1)
            if not job_tuple:
                continue
            job_id, request_payload, client_request_id = job_tuple
            # Skip if canceled before processing
            status_record = await job_store.get_status(job_id)
            if status_record and status_record.get("status") == "cancelled":
                continue
            await job_store.increment_active()
            active_incremented = True
            await job_store.update_status(job_id, "running", started_ts=time.time())
            chat_request = ChatRequest.model_validate(request_payload)
            result_payload = await process_chat_request(
                chat_request,
                job_id=job_id,
                client_request_id=client_request_id,
                completions_url=completions_url,
            )
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
                    "response_metadata": result_payload.get("usage"),
                    "timestamp": time.time(),
                }
            )
        except Exception as exc:
            logger.exception("Worker error")
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
        worker_task, \
        metrics_task, \
        cleanup_task
    http_client = await create_http_client()
    job_store = InMemoryJobStore(settings)
    metrics_sampler = MetricsSampler(settings, job_store, http_client)
    worker_task = asyncio.create_task(worker_loop())
    metrics_task = asyncio.create_task(metrics_sampler.run())
    cleanup_task = asyncio.create_task(cleanup_loop())
    try:
        yield
    finally:
        for task in (worker_task, metrics_task, cleanup_task):
            if task:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
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
        encrypted_content_b64 = message.get("content", "")
        if isinstance(encrypted_content_b64, str):
            encrypted_content_bytes = base64.b64decode(encrypted_content_b64)
            decrypted_content = request_fernet.decrypt(encrypted_content_bytes).decode(
                "utf-8"
            )
            decrypted_messages.append(
                {"role": message.get("role", "user"), "content": decrypted_content}
            )

    formatted_prompt_for_tokenisation = format_prompt_for_llm(
        decrypted_messages, request.model
    )
    needs_two_step = False
    structured_outputs = None

    if request.extra_body:
        extra = request.extra_body
        if "structured_outputs" in extra:
            structured_outputs = extra["structured_outputs"]
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
    guided_request_1 = {
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
            message["content"] = base64.b64encode(encrypted_content).decode("ascii")
            if "thinking" in message and message["thinking"]:
                thinking_bytes = message["thinking"].encode("utf-8")
                encrypted_thinking = request_fernet.encrypt(thinking_bytes)
                message["thinking"] = base64.b64encode(encrypted_thinking).decode(
                    "ascii"
                )
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

    guided_request = {
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
        message["content"] = base64.b64encode(encrypted_content).decode("ascii")
        if "thinking" in message and message["thinking"]:
            thinking_bytes = message["thinking"].encode("utf-8")
            encrypted_thinking = request_fernet.encrypt(thinking_bytes)
            message["thinking"] = base64.b64encode(encrypted_thinking).decode("ascii")

    return isolated_response_data


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
    incoming_requests: asyncio.Queue[dict] = asyncio.Queue()
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
                    try:
                        payload = json.loads(message)
                        msg_type = payload.get("type")

                        if msg_type == "pong":
                            # Pong received - activity already updated above
                            continue
                        elif msg_type == "cancel":
                            # Client wants to cancel a job
                            cancel_job_id = payload.get("request_id")
                            if cancel_job_id and cancel_job_id in pending_jobs:
                                await job_store.cancel_job(cancel_job_id)
                                pending_jobs.pop(cancel_job_id, None)
                            continue
                        else:
                            # Assume it's a completion request
                            await incoming_requests.put(payload)
                    except json.JSONDecodeError as exc:
                        logger.warning("Invalid JSON from client: %s", exc)
                        try:
                            await websocket.send_json(
                                {"type": "error", "error": f"Invalid JSON: {exc}"}
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
            # Process incoming requests (non-blocking check)
            try:
                payload = incoming_requests.get_nowait()
                try:
                    ws_request = CompletionWebSocketRequest.model_validate(payload)
                    job_id = await job_store.enqueue_job(
                        ws_request.request.model_dump(),
                        ws_request.client_request_id,
                    )
                    timeout_seconds = int(
                        ws_request.timeout_seconds or settings.job_timeout_seconds
                    )
                    pending_jobs[job_id] = time.time() + timeout_seconds
                    # Start a waiter task for this job
                    active_waiters[job_id] = asyncio.create_task(
                        result_waiter_task(job_id, timeout_seconds)
                    )
                except Exception as exc:
                    logger.warning("Invalid request payload: %s", exc)
                    try:
                        await websocket.send_json(
                            {"type": "error", "error": f"Invalid payload: {exc}"}
                        )
                    except Exception:
                        pass
            except asyncio.QueueEmpty:
                pass

            # Check for completed job waiters
            completed_waiters = [
                jid for jid, task in active_waiters.items() if task.done()
            ]
            for job_id in completed_waiters:
                task = active_waiters.pop(job_id)
                pending_jobs.pop(job_id, None)
                try:
                    result = task.result()
                    if result is not None:
                        await websocket.send_json(result)
                    else:
                        # Timed out
                        await job_store.cancel_job(job_id)
                        await websocket.send_json(
                            {
                                "type": "timeout",
                                "request_id": job_id,
                                "error": "Request timed out",
                            }
                        )
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
                await job_store.cancel_job(job_id)
                try:
                    await websocket.send_json(
                        {
                            "type": "timeout",
                            "request_id": job_id,
                            "error": "Request timed out",
                        }
                    )
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


@app.get("/dashboard/metrics", response_class=HTMLResponse)
async def dashboard_metrics(request: Request):
    metrics_data = {"timestamp": "n/a"}
    chart_data = {
        "labels": [],
        "avg_gen_throughput": [],
        "gpu_cache_usage": [],
        "memory_usage_percent": [],
        "gpu_utilisation": [],
    }
    if metrics_sampler and metrics_sampler.latest_metrics:
        metrics_data = metrics_sampler.latest_metrics.get("current", metrics_data)
        chart_data = metrics_sampler.latest_metrics.get("chart", chart_data)
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
