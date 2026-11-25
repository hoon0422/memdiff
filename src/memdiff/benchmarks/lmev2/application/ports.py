"""Ports of the LME-V2 benchmark: dataset, reader, judges, run store, remote fetcher."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.lmev2.domain.dataset import LMEV2Question
from memdiff.memory.domain.context import MemoryContextItem
from memdiff.memory.domain.trajectory import Trajectory

T = TypeVar("T")


class Progress(Protocol):
    """Progress display: yields `items` unchanged and reports each one as it is consumed."""

    def track(self, items: Sequence[T], *, desc: str, unit: str) -> Iterable[T]: ...


class LMEV2Dataset(Protocol):
    def load_questions(self, *, domain: str | None = None) -> list[LMEV2Question]: ...

    def load_haystack(self) -> dict[str, list[str]]:
        """question id -> ordered trajectory ids."""
        ...

    def load_trajectories(self, *, allowed_ids: set[str] | None = None) -> dict[str, Trajectory]: ...

    def image_path(self, question: LMEV2Question) -> Path | None:
        """The question screenshot if the question has one and the file exists."""
        ...


class Reader(Protocol):
    def answer(
        self,
        *,
        domain: str,
        question: LMEV2Question,
        context: list[MemoryContextItem],
        image_path: Path | None,
        think: bool | None = None,
    ) -> str:
        """The model's raw response given retrieved memory (and the question screenshot)."""
        ...


class Judge(Protocol):
    def grade(
        self,
        response: str,
        reference: str,
        *,
        question: str,
        parsed_answer: str,
        require_non_empty: bool = True,
        **_: Any,
    ) -> bool: ...


class LMEV2RunStore(Protocol):
    """Where one run's artifacts go (one output directory per run)."""

    run_id: str

    def fresh_db_path(self, policy_name: str, group_idx: int) -> Path:
        """Ledger path for a group; any previous file is removed."""
        ...

    def write_replay(self, policy_name: str, group_idx: int, bundle: ReplayBundle) -> None:
        """One replay per haystack group (each group has its own ledger)."""
        ...

    def write_predictions(self, policy_name: str, records: list[dict[str, Any]]) -> None: ...

    def write_summary(self, policy_name: str, summary: Mapping[str, Any]) -> None: ...


class LMEV2Fetcher(Protocol):
    def fetch(
        self, *, data_root: Path, domain: str, include_question_screenshots: bool
    ) -> dict[str, int]: ...
