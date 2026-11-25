"""dishka providers for the LME-V2 context. Kept apart from the shared providers (#10 runs in parallel)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
from dishka import Provider, Scope, provide

from memdiff.adapters.files.lmev2_dataset import JsonlLMEV2Dataset
from memdiff.adapters.files.lmev2_run_store import JsonLMEV2RunStore
from memdiff.adapters.http.lmev2_fetcher import HuggingFaceLMEV2Fetcher
from memdiff.adapters.llm.judges import build_judges
from memdiff.adapters.llm.lmev2_reader import LMEV2Reader
from memdiff.benchmarks.common.application.ports import UsageReport
from memdiff.benchmarks.common.application.workspace import MemoryWorkspaceFactory
from memdiff.benchmarks.lmev2.application.ports import (
    LMEV2Dataset,
    LMEV2Fetcher,
    LMEV2RunStore,
    Progress,
    Reader,
)
from memdiff.benchmarks.lmev2.application.runner import LMEV2BenchmarkRunner
from memdiff.benchmarks.lmev2.application.scorer import Scorer
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel


class LMEV2FetchProvider(Provider):
    scope = Scope.APP

    @provide
    def fetcher(self) -> Iterator[LMEV2Fetcher]:
        with httpx.Client(timeout=120.0) as client:
            yield HuggingFaceLMEV2Fetcher(client)


class LMEV2Provider(Provider):
    """The runner and its collaborators for one run: data root and output dir are per-run."""

    scope = Scope.APP

    def __init__(self, *, data_root: Path, output_dir: Path, progress: Progress) -> None:
        super().__init__()
        self._data_root = data_root
        self._output_dir = output_dir
        self._progress = progress

    @provide
    def dataset(self) -> LMEV2Dataset:
        return JsonlLMEV2Dataset(self._data_root)

    @provide
    def store(self) -> LMEV2RunStore:
        return JsonLMEV2RunStore(self._output_dir)

    @provide
    def reader(self, chat: ChatModel, prompts: PromptCatalog, settings: Settings) -> Reader:
        return LMEV2Reader(chat, prompts, model=settings.memdiff_chat_model)

    @provide
    def scorer(self, chat: ChatModel, prompts: PromptCatalog, settings: Settings) -> Scorer:
        return Scorer(build_judges(chat, prompts, model=settings.memdiff_judge_model))

    @provide
    def runner(
        self,
        dataset: LMEV2Dataset,
        workspaces: MemoryWorkspaceFactory,
        reader: Reader,
        scorer: Scorer,
        store: LMEV2RunStore,
        usage: UsageReport,
    ) -> LMEV2BenchmarkRunner:
        return LMEV2BenchmarkRunner(
            dataset=dataset,
            workspaces=workspaces,
            reader=reader,
            scorer=scorer,
            store=store,
            usage=usage,
            progress=self._progress,
        )
