"""LME-V2 dataset models, question selection, haystack grouping and trajectory normalization."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from memdiff.memory.domain.trajectory import Trajectory


class LMEV2Question(BaseModel):
    id: str
    domain: str
    environment: str
    question_type: str
    question: str
    image: str | None = None
    answer: str
    eval_function: str


class LMEV2State(BaseModel):
    state_index: int
    step: int | None = None
    url: str
    action: str | None = None
    thought: str | None = None
    accessibility_tree: str = ""
    screenshot: str | None = None


class LMEV2Trajectory(BaseModel):
    id: str
    domain: str
    environment: str
    goal: str
    outcome: str | None = None
    start_url: str
    states: list[LMEV2State] = Field(default_factory=list)


def select_balanced_questions(
    questions: list[LMEV2Question], *, per_type: int
) -> list[LMEV2Question]:
    """Deterministically pick up to `per_type` questions from each question_type."""
    buckets: dict[str, list[LMEV2Question]] = defaultdict(list)
    for q in questions:
        buckets[q.question_type].append(q)
    selected: list[LMEV2Question] = []
    for qtype in sorted(buckets):
        selected.extend(buckets[qtype][:per_type])
    return selected


def select_questions(
    questions: list[LMEV2Question], *, balanced_per_type: int | None, limit: int | None
) -> list[LMEV2Question]:
    """`balanced_per_type` wins over `limit`, as in the original runner."""
    if balanced_per_type is not None:
        return select_balanced_questions(questions, per_type=balanced_per_type)
    if limit is not None:
        return questions[:limit]
    return questions


def needed_trajectory_ids(
    questions: list[LMEV2Question], haystack: Mapping[str, list[str]]
) -> set[str]:
    return {tid for q in questions for tid in haystack[q.id]}


def group_by_haystack(
    questions: list[LMEV2Question], haystack: Mapping[str, list[str]]
) -> list[tuple[tuple[str, ...], list[LMEV2Question]]]:
    """Questions sharing one ordered haystack form a group, in first-occurrence order.

    A question id missing from the haystack raises KeyError.
    """
    groups: dict[tuple[str, ...], list[LMEV2Question]] = defaultdict(list)
    for q in questions:
        groups[tuple(haystack[q.id])].append(q)
    return list(groups.items())


def _stringify_nulls(item: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(item, dict):
        return item
    return {k: ("None" if k in keys and v is None else v) for k, v in item.items()}


def trajectory_from_record(record: Mapping[str, Any]) -> Trajectory:
    """Raw jsonl record -> `Trajectory`, keeping the old raw-dict semantics.

    The legacy policies did `str(record.get("outcome", "unknown"))`, so a key that is present
    with a null value rendered as "None" (real LME-V2 data has null outcomes), while `Trajectory`
    rejects None. Stringify explicit nulls on the fields the policies print.
    """
    data = _stringify_nulls(record, ("outcome", "goal"))
    if isinstance(data.get("states"), list):
        data["states"] = [_stringify_nulls(s, ("url", "accessibility_tree")) for s in data["states"]]
    return Trajectory.model_validate(data)
