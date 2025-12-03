import asyncio
import json
import time
from typing import Any, Dict, Optional
from uuid import uuid4

from redis.asyncio import Redis

from .config import Settings


class RedisJobStore:
    def __init__(self, redis: Redis, settings: Settings):
        self.redis = redis
        self.settings = settings

    def job_key(self, job_id: str) -> str:
        return f"{self.settings.job_result_key_prefix}:{job_id}"

    def result_list_key(self, job_id: str) -> str:
        return f"{self.job_key(job_id)}:out"

    @property
    def active_counter_key(self) -> str:
        return f"{self.settings.job_result_key_prefix}:active_count"

    async def enqueue_job(
        self,
        request_payload: Dict[str, Any],
        client_request_id: Optional[str],
    ) -> str:
        job_id = str(uuid4())
        job_key = self.job_key(job_id)
        record = {
            "status": "pending",
            "request": json.dumps(request_payload),
            "client_request_id": client_request_id or "",
            "created_ts": f"{time.time():.6f}",
        }
        pipe = self.redis.pipeline()
        pipe.hset(job_key, mapping=record)
        pipe.expire(job_key, self.settings.request_ttl_seconds)
        pipe.rpush(self.settings.job_queue_key, job_id)
        await pipe.execute()
        return job_id

    async def wait_for_result(self, job_id: str, timeout: int) -> Dict[str, Any]:
        result_key = self.result_list_key(job_id)
        brpop_result = await self.redis.brpop(result_key, timeout=timeout)
        if not brpop_result:
            raise asyncio.TimeoutError(f"Timed out waiting for job {job_id}")
        _, payload = brpop_result
        return json.loads(payload)

    async def publish_result(self, job_id: str, payload: Dict[str, Any]) -> None:
        result_key = self.result_list_key(job_id)
        pipe = self.redis.pipeline()
        pipe.rpush(result_key, json.dumps(payload))
        pipe.expire(result_key, self.settings.request_ttl_seconds)
        await pipe.execute()

    async def update_status(self, job_id: str, status: str, **extra: Any) -> None:
        job_key = self.job_key(job_id)
        mapping = {"status": status}
        mapping.update({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in extra.items()})
        pipe = self.redis.pipeline()
        pipe.hset(job_key, mapping=mapping)
        pipe.expire(job_key, self.settings.request_ttl_seconds)
        await pipe.execute()

    async def increment_active(self) -> None:
        await self.redis.incr(self.active_counter_key)

    async def decrement_active(self) -> None:
        value = await self.redis.decr(self.active_counter_key)
        if value < 0:
            await self.redis.set(self.active_counter_key, 0)

