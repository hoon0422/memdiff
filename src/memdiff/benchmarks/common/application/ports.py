"""Ports shared by both benchmark tracks."""

from __future__ import annotations

from typing import Protocol


class UsageReport(Protocol):
    """Lifetime LLM usage of the run (shared by every policy)."""

    def as_dict(self) -> dict[str, int]: ...
