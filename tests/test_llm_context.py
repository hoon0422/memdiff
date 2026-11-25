"""LLM context behavior at its one seam: the OpenAI-compatible SDK client, swapped for a fake."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dishka import Container, Provider, Scope, provide
from openai import OpenAI

from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.ports import ChatModel, EmbeddingModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest
from memdiff.llm.domain.usage import UsageStats


class FakeSDK:
    """Records every request; answers deterministically."""

    def __init__(self, reply: str = "ok-response") -> None:
        self.reply = reply
        self.chat_calls: list[dict[str, Any]] = []
        self.embedding_calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)

    def _chat(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
        )

    def _embed(self, **kwargs: Any) -> Any:
        self.embedding_calls.append(kwargs)
        inputs = kwargs["input"]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[3.0, 4.0, 0.0, 0.0]) for _ in inputs],
            usage=SimpleNamespace(total_tokens=len(inputs) * 5),
        )


def _container(tmp_path: Path, sdk: FakeSDK, **overrides: Any) -> Container:
    fields: dict[str, Any] = {"llm_base_url": "http://127.0.0.1:11434/v1", **overrides}
    settings = Settings(
        _env_file=None, embedding_batch_size=2, memdiff_cache_dir=tmp_path, **fields
    )

    class FakeSDKProvider(Provider):
        scope = Scope.APP

        @provide(override=True)
        def sdk(self) -> OpenAI:
            return sdk  # type: ignore[return-value]

    return build_container(settings, FakeSDKProvider())


@pytest.fixture
def sdk() -> FakeSDK:
    return FakeSDK()


@pytest.fixture
def container(tmp_path: Path, sdk: FakeSDK) -> Iterator[Container]:
    c = _container(tmp_path, sdk)
    yield c
    c.close()


HELLO = [ChatMessage(role="user", content="hello")]


def test_chat_uses_max_tokens_on_ollama_forwards_reasoning_effort_and_caches(
    container: Container, sdk: FakeSDK
) -> None:
    chat = container.get(ChatModel)
    request = ChatRequest(model="qwen2.5:7b", max_tokens=256, think=False)

    first = chat.complete(HELLO, request)
    assert first.text == "ok-response"
    assert len(sdk.chat_calls) == 1
    call = sdk.chat_calls[0]
    assert call["max_tokens"] == 256
    assert "max_completion_tokens" not in call
    assert call["extra_body"] == {"reasoning_effort": "none"}
    assert call["messages"] == [{"role": "user", "content": "hello"}]

    assert chat.complete(HELLO, request).text == "ok-response"
    assert len(sdk.chat_calls) == 1

    usage = container.get(UsageStats)
    assert (usage.chat_calls, usage.chat_cache_hits) == (1, 1)
    assert (usage.prompt_tokens, usage.completion_tokens) == (10, 4)


def test_embeddings_are_batched_normalized_and_cached(container: Container, sdk: FakeSDK) -> None:
    embedder = container.get(EmbeddingModel)

    texts = ["a", "b", "c", "d", "e"]
    result = embedder.embed(texts, model="nomic-embed-text")
    assert len(result.vectors) == 5
    assert [len(c["input"]) for c in sdk.embedding_calls] == [2, 2, 1]  # batch size 2
    assert result.vectors[0].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0])  # L2-normalized

    again = embedder.embed(texts, model="nomic-embed-text")
    assert len(again.vectors) == 5
    assert len(sdk.embedding_calls) == 3
    assert container.get(UsageStats).embedding_cache_hits == 5


def test_embedding_cache_is_per_model_and_dedupes_repeated_texts(
    container: Container, sdk: FakeSDK
) -> None:
    embedder = container.get(EmbeddingModel)
    embedder.embed(["a"], model="m1")
    embedder.embed(["a"], model="m2")
    assert len(sdk.embedding_calls) == 2
    assert len(embedder.embed(["a", "a"], model="m1").vectors) == 2
    assert len(sdk.embedding_calls) == 2


def test_chat_cache_key_reflects_every_request_input(container: Container, sdk: FakeSDK) -> None:
    chat = container.get(ChatModel)
    base = ChatRequest(model="m", max_tokens=64)
    chat.complete(HELLO, base)
    variants = [
        (HELLO, base.model_copy(update={"model": "other"})),
        ([ChatMessage(role="user", content="bye")], base),
        (HELLO, base.model_copy(update={"temperature": 0.7})),
        (HELLO, base.model_copy(update={"max_tokens": 65})),
        (HELLO, base.model_copy(update={"response_format": {"type": "json_object"}})),
        (HELLO, base.model_copy(update={"extra_body": {"top_k": 1}})),
        (HELLO, base.model_copy(update={"think": True})),
    ]
    for i, (messages, request) in enumerate(variants, start=2):
        chat.complete(messages, request)
        assert len(sdk.chat_calls) == i, f"variant {i - 1} reused a stale cache entry"
    chat.complete(HELLO, base)  # unchanged request still hits
    assert len(sdk.chat_calls) == len(variants) + 1


def test_chat_cache_key_reflects_base_url(tmp_path: Path) -> None:
    sdk = FakeSDK()
    request = ChatRequest(model="m")
    for url in ("http://127.0.0.1:11434/v1", "http://127.0.0.1:9999/v1"):
        c = _container(tmp_path, sdk, llm_base_url=url)
        c.get(ChatModel).complete(HELLO, request)
        c.close()
    assert len(sdk.chat_calls) == 2


def test_empty_chat_response_is_not_cached(tmp_path: Path) -> None:
    sdk = FakeSDK(reply="  ")
    c = _container(tmp_path, sdk)
    chat = c.get(ChatModel)
    request = ChatRequest(model="m")
    assert chat.complete(HELLO, request).text == ""
    chat.complete(HELLO, request)
    assert len(sdk.chat_calls) == 2
    c.close()


def test_cache_persists_across_containers(tmp_path: Path) -> None:
    sdk = FakeSDK()
    request = ChatRequest(model="m")
    for _ in range(2):
        c = _container(tmp_path, sdk)
        assert c.get(ChatModel).complete(HELLO, request).text == "ok-response"
        c.get(EmbeddingModel).embed(["a"], model="e")
        c.close()
    assert len(sdk.chat_calls) == 1
    assert len(sdk.embedding_calls) == 1


def test_native_openai_uses_max_completion_tokens_and_ignores_think(tmp_path: Path) -> None:
    sdk = FakeSDK()
    c = _container(tmp_path, sdk, llm_base_url=None, openai_api_key="sk-test")
    c.get(ChatModel).complete(HELLO, ChatRequest(model="gpt", max_tokens=99, think=False))
    call = sdk.chat_calls[0]
    assert call["max_completion_tokens"] == 99
    assert "max_tokens" not in call
    assert "extra_body" not in call
    c.close()
