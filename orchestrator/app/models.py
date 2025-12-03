from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: str = Field(default="Qwen3-30B-A3B-Thinking-2507-FP8")
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 128
    stop: Optional[List[str]] = None
    extra_body: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    repetition_penalty: Optional[float] = 1.1


class EmbeddingRequest(BaseModel):
    texts: Optional[List[str]] = None
    input: Optional[List[str]] = None
    model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    encoding_format: Optional[str] = "float"

    @property
    def input_texts(self) -> List[str]:
        if self.texts is not None:
            return self.texts
        if self.input is not None:
            return self.input
        raise ValueError("Either 'texts' or 'input' must be provided")


class CompletionWebSocketRequest(BaseModel):
    request: ChatRequest
    client_request_id: Optional[str] = None
    timeout_seconds: Optional[int] = None

