import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4
from orchestrator.app.config import Settings


class InMemoryJobStore:
    """Ephemeral queue/results store for single-container deployments."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue: asyncio.Queue[Tuple[str, Dict[str, Any], Optional[str]]] = (
            asyncio.Queue()
        )
        self.results: Dict[str, Dict[str, Any]] = {}
        self.status: Dict[str, Dict[str, Any]] = {}
        self.result_waiters: Dict[str, asyncio.Future] = {}
        self.canceled: set[str] = set()
        self.active_jobs = 0
        self.websocket_connections = 0
        self.lock = asyncio.Lock()

    async def enqueue_job(
        self,
        request_payload: Dict[str, Any],
        client_request_id: Optional[str],
    ) -> str:
        job_id = str(uuid4())
        created_ts = time.time()
        async with self.lock:
            self.status[job_id] = {
                "status": "pending",
                "request": request_payload,
                "client_request_id": client_request_id or "",
                "created_ts": created_ts,
            }
        await self.queue.put((job_id, request_payload, client_request_id))
        return job_id

    async def get_next_job(
        self, timeout_seconds: float = 1.0
    ) -> Optional[Tuple[str, Dict[str, Any], Optional[str]]]:
        try:
            while True:
                item = await asyncio.wait_for(self.queue.get(), timeout=timeout_seconds)
                job_id = item[0]
                async with self.lock:
                    if job_id in self.canceled:
                        continue
                    meta = self.status.get(job_id, {})
                    created_ts = meta.get("created_ts", 0.0)
                if (
                    created_ts
                    and (time.time() - created_ts) > self.settings.job_timeout_seconds
                ):
                    # drop stale job
                    await self.update_status(job_id, "expired")
                    await self.publish_result(
                        job_id,
                        {
                            "status": "error",
                            "request_id": job_id,
                            "error": "Job expired before processing",
                        },
                    )
                    continue
                return item
        except asyncio.TimeoutError:
            return None

    async def wait_for_result(self, job_id: str, timeout: int) -> Dict[str, Any]:
        async with self.lock:
            if job_id in self.results:
                return self.results.pop(job_id)
            waiter = self.result_waiters.get(job_id)
            if not waiter:
                waiter = asyncio.get_running_loop().create_future()
                self.result_waiters[job_id] = waiter
        return await asyncio.wait_for(waiter, timeout=timeout)

    async def publish_result(self, job_id: str, payload: Dict[str, Any]) -> None:
        waiter: Optional[asyncio.Future]
        async with self.lock:
            if job_id in self.canceled:
                return
            self.results[job_id] = payload
            waiter = self.result_waiters.pop(job_id, None)
        if waiter and not waiter.done():
            waiter.set_result(payload)

    async def update_status(self, job_id: str, status: str, **extra: Any) -> None:
        async with self.lock:
            current = self.status.get(job_id, {})
            current.update({"status": status})
            current.update(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in extra.items()
                }
            )
            self.status[job_id] = current

    async def increment_active(self) -> None:
        async with self.lock:
            self.active_jobs += 1

    async def decrement_active(self) -> None:
        async with self.lock:
            self.active_jobs = max(0, self.active_jobs - 1)

    async def increment_ws_connections(self) -> None:
        async with self.lock:
            self.websocket_connections += 1

    async def decrement_ws_connections(self) -> None:
        async with self.lock:
            self.websocket_connections = max(0, self.websocket_connections - 1)

    async def get_counters(self) -> Dict[str, int]:
        async with self.lock:
            return {
                "queue_depth": self.queue.qsize(),
                "active_jobs": self.active_jobs,
                "websocket_connections": self.websocket_connections,
            }

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            return self.status.get(job_id)

    async def consume_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        async with self.lock:
            result = self.results.pop(job_id, None)
            self.status.pop(job_id, None)
            self.canceled.discard(job_id)
            waiter = self.result_waiters.pop(job_id, None)
            if waiter and not waiter.done():
                waiter.cancel()
        return result

    async def cancel_job(self, job_id: str) -> None:
        async with self.lock:
            self.canceled.add(job_id)
            self.status[job_id] = {"status": "cancelled", "created_ts": time.time()}
            waiter = self.result_waiters.pop(job_id, None)
            if waiter and not waiter.done():
                waiter.set_result(
                    {
                        "status": "cancelled",
                        "request_id": job_id,
                        "error": "Request cancelled",
                    }
                )
            self.results.pop(job_id, None)

    async def cleanup_expired(self, max_age_seconds: int) -> Dict[str, int]:
        """Remove stale jobs/results to keep memory bounded."""
        now = time.time()
        removed_status = 0
        removed_results = 0
        removed_waiters = 0
        async with self.lock:
            stale_ids = [
                job_id
                for job_id, meta in self.status.items()
                if (now - meta.get("created_ts", now)) > max_age_seconds
            ]
            for job_id in stale_ids:
                self.status.pop(job_id, None)
                if job_id in self.results:
                    self.results.pop(job_id, None)
                    removed_results += 1
                waiter = self.result_waiters.pop(job_id, None)
                if waiter and not waiter.done():
                    waiter.set_exception(asyncio.TimeoutError("Job expired"))
                    removed_waiters += 1
                removed_status += 1
        return {
            "removed_status": removed_status,
            "removed_results": removed_results,
            "removed_waiters": removed_waiters,
        }
