"""ReplayBundle and its nested pydantic models stay 1:1 with web/src/types/replay.ts."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from memdiff.audit.domain.replay import ReplayBundle, ReplayQuestionComparison
from memdiff.audit.domain.timeline import BeliefEdge, TimelineEvent
from memdiff.memory.domain.belief import Belief, Provenance
from memdiff.memory.domain.context import MemoryStats

TS = (Path(__file__).parents[1] / "web/src/types/replay.ts").read_text()


def _ts_fields(name: str) -> set[str]:
    body = re.search(rf"export interface {name} \{{(.*?)\n\}}", TS, re.S)
    assert body, name
    return set(re.findall(r"^  (\w+)\??:", body.group(1), re.M))


@pytest.mark.parametrize(
    "model",
    [Provenance, Belief, BeliefEdge, TimelineEvent, MemoryStats, ReplayQuestionComparison, ReplayBundle],
)
def test_ts_interface_has_exactly_the_python_fields(model: type) -> None:
    assert _ts_fields(model.__name__) == set(model.model_fields)
