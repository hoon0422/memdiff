from __future__ import annotations

from pathlib import Path
from typing import Protocol

from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.drift_sim.domain.prediction import DriftPrediction


class DriftReader(Protocol):
    """Answers one query from the retrieved memory texts; returns the raw model response."""

    def answer(self, question: str, memory_texts: list[str], *, think: bool | None) -> str: ...


class UsageReport(Protocol):
    """Lifetime LLM usage of the run (shared by every policy)."""

    def as_dict(self) -> dict[str, int]: ...


class DriftRunStore(Protocol):
    """Run directory: one SQLite ledger and one predictions file per policy, plus the summary."""

    def db_path(self, run_id: str, policy: str) -> Path:
        """Where the ledger lives (its directory is created). Never creates the file."""
        ...

    def db_exists(self, run_id: str, policy: str) -> bool: ...

    def discard_db(self, run_id: str, policy: str) -> None: ...

    def load_predictions(self, run_id: str, policy: str) -> list[DriftPrediction] | None:
        """`None` when the policy has no predictions file."""
        ...

    def save_predictions(
        self, run_id: str, policy: str, predictions: list[DriftPrediction]
    ) -> None: ...

    def save_replay(self, run_id: str, policy: str, bundle: ReplayBundle) -> None: ...

    def save_summary(self, run_id: str, summary: dict[str, object]) -> None: ...
