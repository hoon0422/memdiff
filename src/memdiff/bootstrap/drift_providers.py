"""DriftSim wiring. Passed to `build_container(settings, DriftSimProvider(runs_dir))` by its CLI."""

from __future__ import annotations

from pathlib import Path

from dishka import Provider, Scope, provide

from memdiff.adapters.files.drift_run_store import FileDriftRunStore
from memdiff.adapters.llm.drift_reader import LLMDriftReader
from memdiff.benchmarks.common.application.workspace import MemoryWorkspaceFactory
from memdiff.benchmarks.drift_sim.application.ports import DriftReader, DriftRunStore, UsageReport
from memdiff.benchmarks.drift_sim.application.runner import DriftBenchmarkRunner
from memdiff.bootstrap.settings import Settings
from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel
from memdiff.llm.domain.usage import UsageStats


class DriftSimProvider(Provider):
    scope = Scope.APP

    def __init__(self, runs_dir: Path) -> None:
        super().__init__()
        self._runs_dir = runs_dir

    @provide
    def run_store(self) -> DriftRunStore:
        return FileDriftRunStore(self._runs_dir)

    @provide
    def reader(self, chat: ChatModel, prompts: PromptCatalog, settings: Settings) -> DriftReader:
        return LLMDriftReader(chat, prompts, model=settings.memdiff_chat_model)

    @provide
    def usage(self, usage: UsageStats) -> UsageReport:
        return usage

    @provide
    def runner(
        self,
        workspaces: MemoryWorkspaceFactory,
        reader: DriftReader,
        runs: DriftRunStore,
        usage: UsageReport,
    ) -> DriftBenchmarkRunner:
        return DriftBenchmarkRunner(workspaces=workspaces, reader=reader, runs=runs, usage=usage)
