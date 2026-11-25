from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from openai import OpenAI

from memdiff.config import Settings, get_settings


@dataclass
class UsageTracker:
    chat_calls: int = 0
    chat_cache_hits: int = 0
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "chat_calls": self.chat_calls,
            "chat_cache_hits": self.chat_cache_hits,
            "embedding_calls": self.embedding_calls,
            "embedding_cache_hits": self.embedding_cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens + self.embedding_tokens,
        }


class CachedLLMClient:
    """OpenAI-compatible LLM + embedding client (Ollama default, OpenAI optional) with SQLite cache."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.usage = UsageTracker()
        self._client: Any | None = client

        cache_dir = Path(self.settings.memdiff_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_db = sqlite3.connect(cache_dir / "llm_cache.db")
        self._cache_db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_cache (
                cache_key TEXT PRIMARY KEY,
                response_text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_cache (
                cache_key TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            );
            """
        )
        self._cache_db.commit()

    def _get_client(self) -> Any:
        if self._client is None:
            base_url = self.settings.effective_base_url
            api_key = self.settings.effective_api_key
            if not base_url and not api_key:
                raise RuntimeError(
                    "Set LLM_BASE_URL (e.g. Ollama) or OPENAI_API_KEY in .env."
                )
            kwargs: dict[str, Any] = {
                "api_key": api_key or "ollama",
                "timeout": self.settings.llm_timeout_seconds,
            }
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def list_ollama_models(self, *, timeout: float = 5.0) -> list[str]:
        """Query Ollama's `/api/tags` endpoint and return installed model names."""
        root = self.settings.ollama_root_url
        if not root:
            return []
        resp = httpx.get(f"{root}/api/tags", timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        return [str(m["name"]) for m in payload.get("models", []) if isinstance(m, dict) and "name" in m]

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 1024,
        response_format: dict[str, Any] | None = None,
        think: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> str:
        target_model = model or self.settings.memdiff_chat_model
        merged_extra_body = dict(extra_body or {})
        if think is not None and self.settings.effective_base_url:
            merged_extra_body.setdefault("reasoning_effort", "medium" if think else "none")
        key = self._hash_payload(
            {
                "base_url": self.settings.effective_base_url,
                "model": target_model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "response_format": response_format,
                "extra_body": merged_extra_body or None,
            }
        )
        if use_cache:
            row = self._cache_db.execute(
                "SELECT response_text FROM chat_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row is not None:
                self.usage.chat_cache_hits += 1
                return str(row[0])

        req: dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        # Ollama and other local OpenAI-compatible servers use `max_tokens` reliably
        if self.settings.effective_base_url:
            req["max_tokens"] = max_completion_tokens
        else:
            req["max_completion_tokens"] = max_completion_tokens
        if response_format is not None:
            req["response_format"] = response_format
        if merged_extra_body:
            req["extra_body"] = merged_extra_body

        resp = self._get_client().chat.completions.create(**req)
        self.usage.chat_calls += 1
        if getattr(resp, "usage", None) is not None:
            self.usage.prompt_tokens += int(getattr(resp.usage, "prompt_tokens", 0) or 0)
            self.usage.completion_tokens += int(getattr(resp.usage, "completion_tokens", 0) or 0)
        text = (resp.choices[0].message.content or "").strip()

        if use_cache and text:
            self._cache_db.execute(
                "INSERT OR REPLACE INTO chat_cache (cache_key, response_text) VALUES (?, ?)",
                (key, text),
            )
            self._cache_db.commit()
        return text

    def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        use_cache: bool = True,
    ) -> list[np.ndarray]:
        if not texts:
            return []
        target_model = model or self.settings.memdiff_embedding_model
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        keys: list[str] = []

        for idx, text in enumerate(texts):
            key = self._hash_payload(
                {
                    "base_url": self.settings.effective_base_url,
                    "model": target_model,
                    "text": text,
                }
            )
            keys.append(key)
            if use_cache:
                row = self._cache_db.execute(
                    "SELECT vector FROM embedding_cache WHERE cache_key = ?", (key,)
                ).fetchone()
                if row is not None:
                    self.usage.embedding_cache_hits += 1
                    results[idx] = np.frombuffer(row[0], dtype=np.float32).copy()
                    continue
            missing_indices.append(idx)
            missing_texts.append(text)

        batch_size = max(1, int(self.settings.embedding_batch_size))
        for start in range(0, len(missing_texts), batch_size):
            batch_texts = missing_texts[start : start + batch_size]
            batch_indices = missing_indices[start : start + batch_size]
            resp = self._get_client().embeddings.create(model=target_model, input=batch_texts)
            self.usage.embedding_calls += 1
            if getattr(resp, "usage", None) is not None:
                self.usage.embedding_tokens += int(getattr(resp.usage, "total_tokens", 0) or 0)
            for idx, emb_obj in zip(batch_indices, resp.data, strict=True):
                vec = np.asarray(emb_obj.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                results[idx] = vec
                if use_cache:
                    self._cache_db.execute(
                        "INSERT OR REPLACE INTO embedding_cache (cache_key, vector) VALUES (?, ?)",
                        (keys[idx], vec.tobytes()),
                    )
        if missing_texts and use_cache:
            self._cache_db.commit()

        return [v for v in results if v is not None]

    def close(self) -> None:
        self._cache_db.close()


# Backwards-compatible alias
CachedOpenAIClient = CachedLLMClient
