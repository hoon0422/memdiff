from __future__ import annotations

from typing import Protocol

import numpy as np

from memdiff.llm.domain.embeddings import EmbeddingResult
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest


class ModelLister(Protocol):
    """Lists the model names the LLM backend has installed."""

    def list_models(self) -> list[str]: ...


class ChatModel(Protocol):
    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion: ...


class EmbeddingModel(Protocol):
    def embed(self, texts: list[str], *, model: str) -> EmbeddingResult: ...


class LLMResponseCache(Protocol):
    """Persistent key/value store for chat text and embedding vectors."""

    def get_chat(self, key: str) -> str | None: ...

    def put_chat(self, key: str, text: str) -> None: ...

    def get_embedding(self, key: str) -> np.ndarray | None: ...

    def put_embeddings(self, items: dict[str, np.ndarray]) -> None: ...
