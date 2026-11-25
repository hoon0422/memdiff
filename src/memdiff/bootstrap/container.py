from __future__ import annotations

from dishka import Container, make_container

from memdiff.bootstrap.providers import ConfigProvider, LLMProvider, PromptProvider
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog


def build_container(settings: Settings) -> Container:
    """Composition root: settings enter as a dishka context, everything else is derived."""
    container = make_container(
        ConfigProvider(), LLMProvider(), PromptProvider(), context={Settings: settings}
    )
    try:
        container.get(PromptCatalog)  # eager: a bad prompt fails startup, not the first LLM call
    except Exception:
        container.close()
        raise
    return container
