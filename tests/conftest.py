from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest


@pytest.fixture
def deterministic_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace random belief/chunk ids and wall-clock event timestamps with counters."""
    ids = itertools.count(1)
    monkeypatch.setattr(
        "memdiff.core.store.uuid",
        SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=f"{next(ids):010x}")),
    )
    ticks = itertools.count()

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def]
            return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=next(ticks))

    monkeypatch.setattr("memdiff.core.models.datetime", _Clock)
