from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memdiff.config import Settings
from memdiff.llm import CachedLLMClient


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok-response"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[1.0, 0.0, 0.0, 0.0]) for _ in inputs],
            usage=SimpleNamespace(total_tokens=len(inputs) * 5),
        )


def test_ollama_client_uses_max_tokens_batches_embeddings_and_caches(tmp_path: Path) -> None:
    fake_completions = _FakeCompletions()
    fake_embeddings = _FakeEmbeddings()
    fake_sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=fake_completions),
        embeddings=fake_embeddings,
    )
    settings = Settings(
        llm_base_url="http://127.0.0.1:11434/v1",
        llm_api_key="ollama",
        memdiff_chat_model="qwen2.5:7b",
        memdiff_embedding_model="nomic-embed-text",
        embedding_batch_size=2,
        memdiff_cache_dir=tmp_path,
    )
    client = CachedLLMClient(settings=settings, client=fake_sdk)

    # First chat call hits backend using `max_tokens` (Ollama-compatible)
    out1 = client.chat([{"role": "user", "content": "hello"}], max_completion_tokens=256)
    assert out1 == "ok-response"
    assert len(fake_completions.calls) == 1
    assert fake_completions.calls[0]["max_tokens"] == 256
    assert "max_completion_tokens" not in fake_completions.calls[0]

    # Second identical call hits SQLite cache
    out2 = client.chat([{"role": "user", "content": "hello"}], max_completion_tokens=256)
    assert out2 == "ok-response"
    assert len(fake_completions.calls) == 1
    assert client.usage.chat_cache_hits == 1

    # Embeddings respect batch_size=2 (5 inputs -> 3 calls) and cache on repeat
    vecs = client.embed(["a", "b", "c", "d", "e"])
    assert len(vecs) == 5
    assert len(fake_embeddings.calls) == 3

    vecs_cached = client.embed(["a", "b", "c", "d", "e"])
    assert len(vecs_cached) == 5
    assert len(fake_embeddings.calls) == 3
    assert client.usage.embedding_cache_hits == 5
    client.close()
