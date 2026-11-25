from __future__ import annotations

from memdiff.llm.application.ports import PromptSource
from memdiff.llm.domain.prompt import PromptId, PromptTemplate


class PromptCatalog:
    """Loads and validates every prompt at construction, so startup fails fast."""

    def __init__(self, source: PromptSource) -> None:
        self._templates = {pid: PromptTemplate(pid, source.read(pid.value)) for pid in PromptId}

    def render(self, prompt_id: PromptId, **values: str) -> str:
        return self._templates[prompt_id].render(**values)
