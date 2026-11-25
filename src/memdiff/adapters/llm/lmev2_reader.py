"""LME-V2 Reader: catalog prompts + `ChatModel`; question screenshots become image data URLs."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from memdiff.benchmarks.lmev2.domain.dataset import LMEV2Question
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest
from memdiff.llm.domain.prompt import PromptId
from memdiff.memory.domain.context import MemoryContextItem

_MAX_COMPLETION_TOKENS = 16384  # qwen3.5 thinks by default; keep room for `content`


def _image_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{b64}"


class LMEV2Reader:
    def __init__(self, chat: ChatModel, prompts: PromptCatalog, *, model: str) -> None:
        self._chat = chat
        self._prompts = prompts
        self._model = model

    def answer(
        self,
        *,
        domain: str,
        question: LMEV2Question,
        context: list[MemoryContextItem],
        image_path: Path | None,
        think: bool | None = None,
    ) -> str:
        system_id = (
            PromptId.LMEV2_READER_SYSTEM_WEB
            if domain == "web"
            else PromptId.LMEV2_READER_SYSTEM_ENTERPRISE
        )
        chunks = [c.value for c in context if c.type == "text"]
        user_text = self._prompts.render(
            PromptId.LMEV2_READER_USER,
            memory="\n\n---\n\n".join(chunks) if chunks else "(no retrieved memory)",
            question=question.question,
        )
        user_content: str | list[dict[str, object]] = user_text
        if image_path is not None:
            user_content = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
            ]
        messages = [
            ChatMessage(role="system", content=self._prompts.render(system_id)),
            ChatMessage(role="user", content=user_content),
        ]
        request = ChatRequest(
            model=self._model, temperature=0.0, max_tokens=_MAX_COMPLETION_TOKENS, think=think
        )
        return self._chat.complete(messages, request).text
