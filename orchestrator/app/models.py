from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = Field(..., min_length=1, description="Model name is required.")
    temperature: float = 0.6
    top_p: float = 0.95
    thinking_effort: Literal["none", "low", "medium", "high"] = "medium"
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

    @field_validator("thinking_effort")
    @classmethod
    def normalise_thinking_effort(cls, v: str) -> str:
        allowed = {"none", "low", "medium", "high"}
        value = v.lower()
        if value not in allowed:
            raise ValueError(f"thinking_effort must be one of {sorted(allowed)}")
        return value


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


class OCRRequest(BaseModel):
    """
    OCR request for DeepSeek-OCR model.
    Messages should follow OpenAI chat format with image_url content.
    """

    messages: List[Dict[str, Any]]
    model: str = Field(default="deepseek-ai/DeepSeek-OCR", min_length=1)
    max_tokens: int = 4000
    temperature: float = 0.0
    extra_body: Optional[Dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def ensure_model(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Model name must be provided")
        return v


class OCRWebSocketRequest(BaseModel):
    request: OCRRequest
    client_request_id: Optional[str] = None
    timeout_seconds: Optional[int] = None
