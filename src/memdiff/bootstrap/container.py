from __future__ import annotations

from dishka import Container, make_container

from memdiff.bootstrap.providers import ConfigProvider, LLMProvider
from memdiff.bootstrap.settings import Settings


def build_container(settings: Settings) -> Container:
    """Composition root: settings enter as a dishka context, everything else is derived."""
    return make_container(ConfigProvider(), LLMProvider(), context={Settings: settings})
