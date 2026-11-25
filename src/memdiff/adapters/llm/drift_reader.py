"""LLM `DriftReader`: answers a DriftSim question from retrieved memory notes."""

from __future__ import annotations

from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest
from memdiff.llm.domain.prompt import PromptId

_MAX_COMPLETION_TOKENS = 16384  # qwen3.5 thinks by default; keep room for `content`


class LLMDriftReader:
    def __init__(self, chat: ChatModel, prompts: PromptCatalog, *, model: str) -> None:
        self._chat, self._model = chat, model
        self._system = prompts.render(PromptId.DRIFT_READER_SYSTEM)
        self._prompts = prompts

    def answer(self, question: str, memory_texts: list[str], *, think: bool | None) -> str:
        memory = "\n\n---\n\n".join(memory_texts) if memory_texts else "(no retrieved memory)"
        user = self._prompts.render(PromptId.DRIFT_READER_USER, memory=memory, question=question)
        return self._chat.complete(
            [ChatMessage(role="system", content=self._system), ChatMessage(role="user", content=user)],
            ChatRequest(
                model=self._model, temperature=0.0, max_tokens=_MAX_COMPLETION_TOKENS, think=think
            ),
        ).text
