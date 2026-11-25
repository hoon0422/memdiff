from __future__ import annotations

from typing import Protocol


class ModelLister(Protocol):
    """Lists the model names the LLM backend has installed."""

    def list_models(self) -> list[str]: ...


class PromptSource(Protocol):
    """Reads raw prompt text by its id value (e.g. `memory/belief_extractor.system`)."""

    def read(self, name: str) -> str:
        """Raise `PromptError` if the prompt does not exist."""
        ...
