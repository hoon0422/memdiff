"""dishka providers: one per bounded context. Add new contexts as new Provider classes."""

from __future__ import annotations

from dishka import Provider, Scope, from_context, provide

from memdiff.adapters.llm.openai_compat import OllamaModelLister
from memdiff.adapters.prompts.fs_source import FilesystemPromptSource
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ModelLister, PromptSource


class ConfigProvider(Provider):
    settings = from_context(provides=Settings, scope=Scope.APP)


class LLMProvider(Provider):
    scope = Scope.APP

    @provide
    def model_lister(self, settings: Settings) -> ModelLister:
        return OllamaModelLister(settings.ollama_root_url)


class PromptProvider(Provider):
    scope = Scope.APP

    @provide
    def prompt_source(self, settings: Settings) -> PromptSource:
        return FilesystemPromptSource(settings.memdiff_prompts_dir)

    @provide
    def prompt_catalog(self, source: PromptSource) -> PromptCatalog:
        return PromptCatalog(source)
