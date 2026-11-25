"""Progress implementations that need no outside library."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")


class NoProgress:
    """Reports nothing (tests, non-interactive runs)."""

    def track(self, items: Sequence[T], *, desc: str, unit: str) -> Iterable[T]:
        return items
