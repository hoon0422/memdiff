"""Cache-through decorators: same ports in, same ports out, with usage + cache-hit accounting."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from memdiff.llm.application.ports import ChatModel, EmbeddingModel, LLMResponseCache
from memdiff.llm.domain.embeddings import EmbeddingResult
from memdiff.llm.domain.messages import ChatCompletion, ChatMessage, ChatRequest
from memdiff.llm.domain.usage import UsageStats


def _digest(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chat_cache_key(base_url: str | None, messages: list[ChatMessage], request: ChatRequest) -> str:
    return _digest(
        {
            "base_url": base_url,
            "messages": [m.model_dump() for m in messages],
            "request": request.model_dump(),  # model, temperature, max_tokens, format, think, extra_body
        }
    )


def embedding_cache_key(base_url: str | None, model: str, text: str) -> str:
    return _digest({"base_url": base_url, "model": model, "text": text})


class CachingChatModel:
    def __init__(
        self, inner: ChatModel, cache: LLMResponseCache, usage: UsageStats, *, base_url: str | None
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._usage = usage
        self._base_url = base_url

    def complete(self, messages: list[ChatMessage], request: ChatRequest) -> ChatCompletion:
        key = chat_cache_key(self._base_url, messages, request)
        cached = self._cache.get_chat(key)
        if cached is not None:
            self._usage.chat_cache_hits += 1
            return ChatCompletion(text=cached)

        completion = self._inner.complete(messages, request)
        self._usage.chat_calls += 1
        self._usage.prompt_tokens += completion.prompt_tokens
        self._usage.completion_tokens += completion.completion_tokens
        if completion.text:  # never cache empty output (thinking can exhaust the token cap)
            self._cache.put_chat(key, completion.text)
        return completion


class CachingEmbeddingModel:
    def __init__(
        self,
        inner: EmbeddingModel,
        cache: LLMResponseCache,
        usage: UsageStats,
        *,
        base_url: str | None,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._usage = usage
        self._base_url = base_url

    def embed(self, texts: list[str], *, model: str) -> EmbeddingResult:
        keys = [embedding_cache_key(self._base_url, model, t) for t in texts]
        found = {k: v for k in set(keys) if (v := self._cache.get_embedding(k)) is not None}
        self._usage.embedding_cache_hits += sum(1 for k in keys if k in found)

        missing = [(k, t) for k, t in zip(keys, texts, strict=True) if k not in found]
        if missing:
            result = self._inner.embed([t for _, t in missing], model=model)
            self._usage.embedding_calls += 1
            self._usage.embedding_tokens += result.total_tokens
            fresh = {k: v for (k, _), v in zip(missing, result.vectors, strict=True)}
            self._cache.put_embeddings(fresh)
            found.update(fresh)
        return EmbeddingResult(vectors=[found[k] for k in keys])
