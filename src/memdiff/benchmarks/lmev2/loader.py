from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


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


def load_questions(
    data_root: str | Path,
    *,
    domain: str | None = None,
    question_types: set[str] | None = None,
    limit: int | None = None,
) -> list[LMEV2Question]:
    path = Path(data_root) / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `uv run python scripts/fetch_lmev2_small.py` first."
        )
    rows: list[LMEV2Question] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            item = LMEV2Question.model_validate(json.loads(stripped))
            if domain is not None and item.domain != domain:
                continue
            if question_types is not None and item.question_type not in question_types:
                continue
            rows.append(item)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_haystack(data_root: str | Path, *, tier: str = "small") -> dict[str, list[str]]:
    path = Path(data_root) / "haystacks" / f"lme_v2_{tier}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing haystack file: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): [str(tid) for tid in v] for k, v in raw.items()}


def load_trajectories(
    data_root: str | Path,
    *,
    filename: str = "trajectories_small.jsonl",
    allowed_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    root = Path(data_root)
    path = root / filename
    if not path.exists():
        fallback = root / "trajectories.jsonl"
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(f"Missing trajectories file: {path}")

    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            tid = str(record["id"])
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            out[tid] = record
    return out
