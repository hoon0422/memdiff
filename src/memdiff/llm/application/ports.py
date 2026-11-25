from __future__ import annotations

from typing import Protocol


class ModelLister(Protocol):
    """Lists the model names the LLM backend has installed."""

    def list_models(self) -> list[str]: ...
