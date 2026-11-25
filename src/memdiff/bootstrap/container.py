from __future__ import annotations

from dishka import Container, Provider, make_container

from memdiff.bootstrap.providers import ConfigProvider, LLMProvider, PromptProvider
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog


def build_container(settings: Settings, *overrides: Provider) -> Container:
    """Composition root: settings enter as a dishka context, everything else is derived.

    `overrides` are extra providers (tests, offline demo) whose `@provide(override=True)` wins.
    """
    container = make_container(
        ConfigProvider(), LLMProvider(), PromptProvider(), *overrides, context={Settings: settings}
    )
    try:
        container.get(PromptCatalog)  # eager: a bad prompt fails startup, not the first LLM call
    except Exception:
        container.close()
        raise
    return container
