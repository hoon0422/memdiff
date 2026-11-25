"""LLM `DriftReader`: answers a DriftSim question from retrieved memory notes."""

from __future__ import annotations

from memdiff.adapters.llm.chat import ask, join_memory
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.prompt import PromptId


class LLMDriftReader:
    def __init__(self, chat: ChatModel, prompts: PromptCatalog, *, model: str) -> None:
        self._chat, self._model = chat, model
        self._system = prompts.render(PromptId.DRIFT_READER_SYSTEM)
        self._prompts = prompts

    def answer(self, question: str, memory_texts: list[str], *, think: bool | None) -> str:
        user = self._prompts.render(
            PromptId.DRIFT_READER_USER, memory=join_memory(memory_texts), question=question
        )
        return ask(self._chat, self._system, user, model=self._model, think=think)
