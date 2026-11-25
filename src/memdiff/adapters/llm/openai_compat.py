from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from memdiff.llm.domain.embeddings import EmbeddingResult
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest


class OllamaModelLister:
    """`ModelLister` backed by Ollama's `/api/tags` endpoint."""

    def __init__(self, root_url: str | None, *, timeout: float = 5.0) -> None:
        self._root_url = root_url
        self._timeout = timeout

    def list_models(self) -> list[str]:
        if not self._root_url:
            return []
        resp = httpx.get(f"{self._root_url}/api/tags", timeout=self._timeout)
        resp.raise_for_status()
        payload = resp.json()
        return [
            str(m["name"]) for m in payload.get("models", []) if isinstance(m, dict) and "name" in m
        ]


class OpenAIChatModel:
    """`ChatModel` over an OpenAI-compatible SDK client.

    A set `base_url` means a local server (Ollama): it honors `max_tokens` (not
    `max_completion_tokens`) and takes `think` as `reasoning_effort`.
    """

    def __init__(self, client: Any, *, base_url: str | None) -> None:
        self._client = client
        self._base_url = base_url

    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion:
        extra_body = dict(request.extra_body or {})
        if request.think is not None and self._base_url:
            extra_body.setdefault("reasoning_effort", "medium" if request.think else "none")
        req: dict[str, Any] = {
            "model": request.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": request.temperature,
            "max_tokens" if self._base_url else "max_completion_tokens": request.max_tokens,
        }
        if request.response_format is not None:
            req["response_format"] = request.response_format
        if extra_body:
            req["extra_body"] = extra_body

        resp = self._client.chat.completions.create(**req)
        usage = getattr(resp, "usage", None)
        return ChatCompletion(
            text=(resp.choices[0].message.content or "").strip(),
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )


class OpenAIEmbeddingModel:
    """`EmbeddingModel` over an OpenAI-compatible SDK client: batched, L2-normalized."""

    def __init__(self, client: Any, *, batch_size: int) -> None:
        self._client = client
        self._batch_size = max(1, batch_size)

    def embed(self, texts: list[str], *, model: str) -> EmbeddingResult:
        vectors: list[np.ndarray] = []
        total_tokens = 0
        for start in range(0, len(texts), self._batch_size):
            resp = self._client.embeddings.create(
                model=model, input=texts[start : start + self._batch_size]
            )
            usage = getattr(resp, "usage", None)
            total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            for item in resp.data:
                vec = np.asarray(item.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                vectors.append(vec / norm if norm > 0 else vec)
        return EmbeddingResult(vectors=vectors, total_tokens=total_tokens)
