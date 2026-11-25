from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    # str, or OpenAI content parts (text / image_url dicts) for multimodal prompts
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    """Generation parameters. `think=None` leaves the backend default untouched."""

    model_config = ConfigDict(frozen=True)

    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    response_format: dict[str, Any] | None = None
    think: bool | None = None
    extra_body: dict[str, Any] | None = None


class ChatCompletion(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
