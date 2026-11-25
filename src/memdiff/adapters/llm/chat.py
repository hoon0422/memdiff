"""Shared shape of every LLM call the adapters make: system + user message, temperature 0."""

from __future__ import annotations

from typing import Any

from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest

MAX_COMPLETION_TOKENS = 16384  # qwen3.5 thinks by default; keep room for `content`


def ask(
    chat: ChatModel,
    system: str,
    user: str | list[dict[str, Any]],
    *,
    model: str,
    think: bool | None = None,
) -> str:
    """Request bytes are pinned by the goldens in tests/golden: do not reorder or add fields."""
    return chat.complete(
        [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        ChatRequest(model=model, temperature=0.0, max_tokens=MAX_COMPLETION_TOKENS, think=think),
    ).text


def join_memory(texts: list[str]) -> str:
    """Retrieved memory blocks as the reader prompts expect them."""
    return "\n\n---\n\n".join(texts) if texts else "(no retrieved memory)"
