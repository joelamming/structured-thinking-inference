import asyncio
import json
import logging
import re
import subprocess
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

from redis.asyncio import Redis

from .config import Settings
from .jobs import RedisJobStore


METRIC_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)$'
)


class MetricsSampler:
    def __init__(
        self,
        redis: Redis,
        settings: Settings,
        job_store: RedisJobStore,
        http_client,
        max_points: int = 120,
    ):
        self.redis = redis
        self.settings = settings
        self.job_store = job_store
        self.http_client = http_client
        self.previous_prompt_tokens = 0.0
        self.previous_generation_tokens = 0.0
        self.last_collection_ts = 0.0
        self.chart_labels: deque[str] = deque(maxlen=max_points)
        self.chart_avg_gen_throughput: deque[float] = deque(maxlen=max_points)
        self.chart_gpu_cache_usage: deque[float] = deque(maxlen=max_points)
        self.chart_memory_usage_percent: deque[float] = deque(maxlen=max_points)
        self.chart_gpu_utilisation: deque[float] = deque(maxlen=max_points)

    async def fetch_server_metrics(self) -> Optional[str]:
        try:
            response = await self.http_client.get(self.settings.llm_metrics_url, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    def parse_metrics(self, metrics_str: str) -> Dict[str, Dict[str, Any]]:
        lines = metrics_str.strip().splitlines()
        parsed: Dict[str, Dict[str, Any]] = {}
        for line in lines:
            match = METRIC_LINE_RE.match(line)
            if match:
                name, labels_str, value = match.groups()
                labels = {}
                if labels_str:
                    pairs = labels_str.strip("{}").split(",")
                    for pair in pairs:
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            labels[k.strip()] = v.strip('"')
                parsed[name.strip()] = {"labels": labels, "value": float(value)}
        return parsed

    def compute_throughput(self, parsed: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        prompt_total = parsed.get("vllm:prompt_tokens_total", {}).get("value", 0.0)
        generation_total = parsed.get("vllm:generation_tokens_total", {}).get("value", 0.0)
        now = time.time()
        delta_t = now - self.last_collection_ts if self.last_collection_ts else 0.0

        if delta_t <= 0:
            prompt_throughput = 0.0
            generation_throughput = 0.0
        else:
            prompt_delta = prompt_total - self.previous_prompt_tokens
            if prompt_delta < 0:
                prompt_delta = prompt_total
            prompt_throughput = prompt_delta / delta_t

            generation_delta = generation_total - self.previous_generation_tokens
            if generation_delta < 0:
                generation_delta = generation_total
            generation_throughput = generation_delta / delta_t

        self.previous_prompt_tokens = prompt_total
        self.previous_generation_tokens = generation_total
        self.last_collection_ts = now

        return {
            "vllm:avg_prompt_throughput_toks_per_s": prompt_throughput,
            "vllm:avg_generation_throughput_toks_per_s": generation_throughput,
        }

    async def get_gpu_metrics(self) -> Dict[str, float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._collect_gpu_metrics)

    def _collect_gpu_metrics(self) -> Dict[str, float]:
        try:
            command = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,power.draw,power.limit,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=5,
            )
            output = result.stdout.strip()
            if not output:
                return {}
            values = output.split(", ")
            if len(values) != 5:
                return {}
            memory_used = float(values[0])
            memory_total = float(values[1])
            power_draw = float(values[2])
            power_limit = float(values[3])
            gpu_utilisation = float(values[4])
            memory_usage_percent = (memory_used / memory_total * 100) if memory_total else 0.0
            return {
                "memory_usage_mb": memory_used,
                "memory_total_mb": memory_total,
                "power_draw_w": power_draw,
                "power_limit_w": power_limit,
                "gpu_utilisation_percent": gpu_utilisation,
                "memory_usage_percent": round(memory_usage_percent, 2),
            }
        except Exception:
            return {}

    async def sample_once(self) -> None:
        metrics_text = await self.fetch_server_metrics()
        parsed: Dict[str, Dict[str, Any]] = {}
        if metrics_text:
            parsed = self.parse_metrics(metrics_text)
        throughput = self.compute_throughput(parsed) if parsed else {
            "vllm:avg_prompt_throughput_toks_per_s": 0.0,
            "vllm:avg_generation_throughput_toks_per_s": 0.0,
        }
        gpu_metrics = await self.get_gpu_metrics()
        queue_depth = await self.redis.llen(self.settings.job_queue_key)
        active_jobs_raw = await self.redis.get(self.job_store.active_counter_key)
        active_jobs = int(active_jobs_raw or 0)
        ws_connections_raw = await self.redis.get("llm:ws:connections")
        ws_connections = int(ws_connections_raw or 0)

        current_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        current_metrics = {
            "timestamp": current_timestamp,
            "queue_depth": queue_depth,
            "active_jobs": active_jobs,
            "websocket_connections": ws_connections,
            **throughput,
        }
        for key in (
            "vllm:gpu_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:num_requests_swapped",
            "vllm:cpu_cache_usage_perc",
        ):
            if key in parsed:
                value = parsed[key]["value"]
                if key == "vllm:gpu_cache_usage_perc":
                    value *= 100
                current_metrics[key] = value

        current_metrics.update(gpu_metrics)

        # Update chart series
        self.chart_labels.append(datetime.utcnow().strftime("%H:%M:%S"))
        self.chart_avg_gen_throughput.append(current_metrics.get("vllm:avg_generation_throughput_toks_per_s", 0.0))
        self.chart_gpu_cache_usage.append(current_metrics.get("vllm:gpu_cache_usage_perc", 0.0))
        self.chart_memory_usage_percent.append(current_metrics.get("memory_usage_percent", 0.0))
        self.chart_gpu_utilisation.append(current_metrics.get("gpu_utilisation_percent", 0.0))

        chart_data = {
            "labels": list(self.chart_labels),
            "avg_gen_throughput": list(self.chart_avg_gen_throughput),
            "gpu_cache_usage": list(self.chart_gpu_cache_usage),
            "memory_usage_percent": list(self.chart_memory_usage_percent),
            "gpu_utilisation": list(self.chart_gpu_utilisation),
        }

        pipe = self.redis.pipeline()
        pipe.set("llm:metrics:current", json.dumps(current_metrics))
        pipe.set("llm:metrics:chart", json.dumps(chart_data))
        await pipe.execute()

    async def run(self) -> None:
        interval = max(1, self.settings.metrics_interval_seconds)
        logger = logging.getLogger("custom_vllm.metrics")
        while True:
            try:
                await self.sample_once()
            except Exception as exc:
                logger.warning("Metrics sampler error: %s", exc)
            await asyncio.sleep(interval)

