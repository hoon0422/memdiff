"""dishka providers: one per bounded context. Add new contexts as new Provider classes."""

from __future__ import annotations

from collections.abc import Iterator

from dishka import Provider, Scope, from_context, provide
from openai import OpenAI
from sqlalchemy import Engine, create_engine

from memdiff.adapters.llm.openai_compat import (
    OllamaModelLister,
    OpenAIChatModel,
    OpenAIEmbeddingModel,
)
from memdiff.adapters.persistence.sqlalchemy.llm_cache import SqlLLMResponseCache
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.caching import CachingChatModel, CachingEmbeddingModel
from memdiff.llm.application.ports import (
    ChatModel,
    EmbeddingModel,
    LLMResponseCache,
    ModelLister,
)
from memdiff.llm.domain.usage import UsageStats


class ConfigProvider(Provider):
    settings = from_context(provides=Settings, scope=Scope.APP)


class LLMProvider(Provider):
    scope = Scope.APP

    @provide
    def model_lister(self, settings: Settings) -> ModelLister:
        return OllamaModelLister(settings.ollama_root_url)

    @provide
    def sdk_client(self, settings: Settings) -> OpenAI:
        """Swap point for tests: override this one provider with a fake SDK."""
        base_url, api_key = settings.effective_base_url, settings.effective_api_key
        if not base_url and not api_key:
            raise RuntimeError("Set LLM_BASE_URL (e.g. Ollama) or OPENAI_API_KEY in .env.")
        return OpenAI(
            api_key=api_key or "ollama",
            timeout=settings.llm_timeout_seconds,
            **({"base_url": base_url} if base_url else {}),
        )

    @provide
    def cache_engine(self, settings: Settings) -> Iterator[Engine]:
        settings.memdiff_cache_dir.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{settings.memdiff_cache_dir / 'llm_cache.db'}")
        yield engine
        engine.dispose()

    @provide
    def response_cache(self, engine: Engine) -> LLMResponseCache:
        return SqlLLMResponseCache(engine)

    @provide
    def usage_stats(self) -> UsageStats:
        return UsageStats()

    @provide
    def chat_model(
        self, client: OpenAI, cache: LLMResponseCache, usage: UsageStats, settings: Settings
    ) -> ChatModel:
        base_url = settings.effective_base_url
        return CachingChatModel(
            OpenAIChatModel(client, base_url=base_url), cache, usage, base_url=base_url
        )

    @provide
    def embedding_model(
        self, client: OpenAI, cache: LLMResponseCache, usage: UsageStats, settings: Settings
    ) -> EmbeddingModel:
        return CachingEmbeddingModel(
            OpenAIEmbeddingModel(client, batch_size=settings.embedding_batch_size),
            cache,
            usage,
            base_url=settings.effective_base_url,
        )
