from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = Field(..., min_length=1, description="Model name is required.")
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 128
    stop: Optional[List[str]] = None
    extra_body: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    repetition_penalty: Optional[float] = None

    @field_validator("model")
    @classmethod
    def ensure_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name must be provided")
        return v


class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    input: Optional[List[str]] = None
    model: str = Field(..., min_length=1, description="Model name is required.")
    encoding_format: Optional[str] = "float"

    @property
    def input_texts(self) -> List[str]:
        if self.texts is not None:
            return self.texts
        if self.input is not None:
            return self.input
        raise ValueError("Either 'texts' or 'input' must be provided")

    @field_validator("model")
    @classmethod
    def ensure_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name must be provided")
        return v


class CompletionWebSocketRequest(BaseModel):
    request: ChatRequest
    client_request_id: Optional[str] = None
    timeout_seconds: Optional[int] = None
