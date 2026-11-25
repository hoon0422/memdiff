"""dishka providers: one per bounded context. Add new contexts as new Provider classes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from dishka import Container, Provider, Scope, alias, from_context, provide
from openai import OpenAI
from sqlalchemy import Engine, create_engine

from memdiff.adapters.llm.belief_ops import LLMBeliefExtractor, LLMBeliefReconciler, LLMEmbedder
from memdiff.adapters.llm.openai_compat import (
    OllamaModelLister,
    OpenAIChatModel,
    OpenAIEmbeddingModel,
)
from memdiff.adapters.offline.belief_ops import (
    OfflineBeliefExtractor,
    OfflineBeliefReconciler,
    OfflineEmbedder,
)
from memdiff.adapters.persistence.sqlalchemy.ledger import (
    SqlAuditLog,
    SqlBeliefRepository,
    SqlChunkRepository,
    SqlUnitOfWork,
    make_ledger_engine,
)
from memdiff.adapters.persistence.sqlalchemy.llm_cache import SqlLLMResponseCache
from memdiff.adapters.prompts.fs_source import FilesystemPromptSource
from memdiff.audit.application.ports import AuditLogRepository, BeliefSource
from memdiff.audit.application.recorder import RevisionAuditRecorder
from memdiff.audit.application.replay import ReplayService
from memdiff.benchmarks.common.application.workspace import (
    MemoryWorkspace,
    MemoryWorkspaceFactory,
    WorkspaceSpec,
)
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.caching import CachingChatModel, CachingEmbeddingModel
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import (
    ChatModel,
    EmbeddingModel,
    LLMResponseCache,
    ModelLister,
    PromptSource,
)
from memdiff.llm.domain.usage import UsageStats
from memdiff.memory.application.policies import (
    BeliefRevisionPolicy,
    NaiveRagPolicy,
    RecencyRagPolicy,
    StructuredExtractPolicy,
)
from memdiff.memory.application.policy import MemoryPolicy
from memdiff.memory.application.ports import (
    BeliefExtractor,
    BeliefReconciler,
    BeliefRepository,
    ChunkRepository,
    Embedder,
    RevisionEventSink,
    UnitOfWork,
)


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


class PromptProvider(Provider):
    scope = Scope.APP

    @provide
    def prompt_source(self, settings: Settings) -> PromptSource:
        return FilesystemPromptSource(settings.memdiff_prompts_dir)

    @provide
    def prompt_catalog(self, source: PromptSource) -> PromptCatalog:
        return PromptCatalog(source)


class MemoryProvider(Provider):
    """One ledger + policy per `WorkspaceSpec`; leaving the REQUEST scope releases the DB."""

    scope = Scope.REQUEST

    spec = from_context(provides=WorkspaceSpec, scope=Scope.REQUEST)

    @provide
    def sql_uow(self, spec: WorkspaceSpec) -> Iterator[SqlUnitOfWork]:
        # the ledger Engine is private to the UoW: the APP-level cache Engine owns that type key
        uow = SqlUnitOfWork(make_ledger_engine(spec.db_path))
        yield uow
        uow.close()  # rolls back anything uncommitted, closes the session, disposes the engine

    uow = alias(source=SqlUnitOfWork, provides=UnitOfWork)

    @provide
    def beliefs(self, uow: SqlUnitOfWork) -> BeliefRepository:
        return SqlBeliefRepository(uow)

    belief_source = alias(source=BeliefRepository, provides=BeliefSource)

    @provide
    def chunks(self, uow: SqlUnitOfWork) -> ChunkRepository:
        return SqlChunkRepository(uow)

    @provide
    def audit_log(self, uow: SqlUnitOfWork) -> AuditLogRepository:
        return SqlAuditLog(uow)

    @provide
    def events(self, log: AuditLogRepository) -> RevisionEventSink:
        return RevisionAuditRecorder(log)

    @provide
    def replay(self, beliefs: BeliefSource, log: AuditLogRepository) -> ReplayService:
        return ReplayService(beliefs=beliefs, log=log)

    # LLM-backed by default; OfflineProvider overrides these three. Cheap objects around the
    # APP-level models, so token usage stays shared across workspaces.
    @provide
    def embedder(self, model: EmbeddingModel, settings: Settings) -> Embedder:
        return LLMEmbedder(model, model_name=settings.memdiff_embedding_model)

    @provide
    def extractor(
        self, chat: ChatModel, prompts: PromptCatalog, settings: Settings, spec: WorkspaceSpec
    ) -> BeliefExtractor:
        return LLMBeliefExtractor(chat, prompts, model=settings.memdiff_chat_model, think=spec.think)

    @provide
    def reconciler(
        self, chat: ChatModel, prompts: PromptCatalog, settings: Settings, spec: WorkspaceSpec
    ) -> BeliefReconciler:
        return LLMBeliefReconciler(
            chat, prompts, model=settings.memdiff_chat_model, think=spec.think
        )

    @provide
    def policy(
        self,
        spec: WorkspaceSpec,
        beliefs: BeliefRepository,
        chunks: ChunkRepository,
        embedder: Embedder,
        extractor: BeliefExtractor,
        reconciler: BeliefReconciler,
        events: RevisionEventSink,
        uow: UnitOfWork,
    ) -> MemoryPolicy:
        match spec.policy_name:
            case "naive_rag":
                return NaiveRagPolicy(
                    chunks=chunks, embedder=embedder, uow=uow, max_chunk_chars=spec.max_chunk_chars
                )
            case "recency_rag":
                return RecencyRagPolicy(
                    chunks=chunks,
                    embedder=embedder,
                    uow=uow,
                    recency_decay=spec.recency_decay,
                    max_chunk_chars=spec.max_chunk_chars,
                )
            case "structured_extract":
                return StructuredExtractPolicy(
                    beliefs=beliefs,
                    chunks=chunks,
                    embedder=embedder,
                    extractor=extractor,
                    events=events,
                    uow=uow,
                )
            case "belief_revision":
                return BeliefRevisionPolicy(
                    beliefs=beliefs,
                    chunks=chunks,
                    embedder=embedder,
                    extractor=extractor,
                    reconciler=reconciler,
                    events=events,
                    uow=uow,
                    candidate_top_k=spec.candidate_top_k,
                )
        raise ValueError(
            f"Unknown policy '{spec.policy_name}'. Expected one of: "
            "naive_rag, recency_rag, structured_extract, belief_revision."
        )


class _ContainerWorkspaceFactory:
    """Opens a REQUEST scope per workspace; the container is hidden from runners."""

    def __init__(self, container: Container) -> None:
        self._container = container

    @contextmanager
    def open(self, spec: WorkspaceSpec) -> Iterator[MemoryWorkspace]:
        with self._container(context={WorkspaceSpec: spec}) as request:
            yield MemoryWorkspace(
                policy=request.get(MemoryPolicy),
                beliefs=request.get(BeliefRepository),
                log=request.get(AuditLogRepository),
                replay=request.get(ReplayService),
            )


class WorkspaceProvider(Provider):
    scope = Scope.APP

    @provide
    def workspace_factory(self, container: Container) -> MemoryWorkspaceFactory:
        return _ContainerWorkspaceFactory(container)


class OfflineProvider(Provider):
    """Rule-based embedder/extractor/reconciler for the offline demo (no LLM calls)."""

    scope = Scope.REQUEST

    @provide(override=True)
    def embedder(self) -> Embedder:
        return OfflineEmbedder()

    @provide(override=True)
    def extractor(self) -> BeliefExtractor:
        return OfflineBeliefExtractor()

    @provide(override=True)
    def reconciler(self) -> BeliefReconciler:
        return OfflineBeliefReconciler()
