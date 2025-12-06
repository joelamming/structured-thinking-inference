from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: str
    encryption_key: str
    llm_server_url: str = "http://127.0.0.1:8000/v1/completions"
    llm_metrics_url: str = "http://127.0.0.1:8000/metrics"
    embedding_server_url: str = "http://127.0.0.1:8000/v1/embeddings"
    log_file_path: str = "llm_request_logs.jsonl"
    enable_request_logs: bool = False
    metrics_interval_seconds: int = 5
    job_timeout_seconds: int = 3600
    openapi_version: str = "3.1.1"
    log_level: str = "INFO"
    structural_log_id: Optional[str] = None
    model_config = {
        "env_file_encoding": "utf-8",
        "env_prefix": "ORCH_",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()
