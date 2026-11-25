"""tqdm-backed `Progress` for the CLIs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


class TqdmProgress:
    def track(self, items: Sequence[T], *, desc: str, unit: str) -> Iterable[T]:
        return tqdm(items, desc=desc, unit=unit)
