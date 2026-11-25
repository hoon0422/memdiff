from __future__ import annotations

from dishka import Container, Provider, make_container

from memdiff.bootstrap.providers import ConfigProvider, LLMProvider
from memdiff.bootstrap.settings import Settings


def build_container(settings: Settings, *overrides: Provider) -> Container:
    """Composition root: settings enter as a dishka context, everything else is derived.

    `overrides` are extra providers (tests, offline demo) whose `@provide(override=True)` wins.
    """
    return make_container(ConfigProvider(), LLMProvider(), *overrides, context={Settings: settings})
