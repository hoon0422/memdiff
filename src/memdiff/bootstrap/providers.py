"""dishka providers: one per bounded context. Add new contexts as new Provider classes."""

from __future__ import annotations

from dishka import Provider, Scope, from_context, provide

from memdiff.adapters.llm.openai_compat import OllamaModelLister
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.ports import ModelLister


class ConfigProvider(Provider):
    settings = from_context(provides=Settings, scope=Scope.APP)


class LLMProvider(Provider):
    scope = Scope.APP

    @provide
    def model_lister(self, settings: Settings) -> ModelLister:
        return OllamaModelLister(settings.ollama_root_url)
