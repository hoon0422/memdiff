from __future__ import annotations

from pathlib import Path
from typing import Protocol

from memdiff.audit.domain.replay import ReplayBundle
from memdiff.benchmarks.drift_sim.domain.prediction import DriftPrediction


class DriftReader(Protocol):
    """Answers one query from the retrieved memory texts; returns the raw model response."""

    def answer(self, question: str, memory_texts: list[str], *, think: bool | None) -> str: ...


class DriftRunStore(Protocol):
    """One run directory: a SQLite ledger and a predictions file per policy, plus the summary."""

    run_id: str

    def db_path(self, policy: str) -> Path:
        """Where the ledger lives. Never creates the file."""
        ...

    def db_exists(self, policy: str) -> bool: ...

    def discard_db(self, policy: str) -> None: ...

    def load_predictions(self, policy: str) -> list[DriftPrediction] | None:
        """`None` when the policy has no predictions file."""
        ...

    def save_predictions(self, policy: str, predictions: list[DriftPrediction]) -> None: ...

    def save_replay(self, policy: str, bundle: ReplayBundle) -> None: ...

    def save_summary(self, summary: dict[str, object]) -> None: ...
