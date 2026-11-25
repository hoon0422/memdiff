"""Route an eval_function spec to a deterministic matcher or an LLM judge."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from memdiff.benchmarks.lmev2.application.ports import Judge
from memdiff.benchmarks.lmev2.domain.scoring.matchers import MATCHERS
from memdiff.benchmarks.lmev2.domain.scoring.spec import EvalSpec


class ScoredQuestion(Protocol):
    question: str
    answer: str


class Scorer:
    def __init__(self, judges: Mapping[str, Judge]) -> None:
        self._judges = judges

    def score(self, spec: str, raw: str, parsed: str, question: ScoredQuestion) -> bool:
        """Judges grade the RAW response; deterministic matchers grade the PARSED answer."""
        parsed_spec = EvalSpec.parse(spec)
        if parsed_spec.is_judge:
            return bool(
                self._judges[parsed_spec.name].grade(
                    raw,
                    question.answer,
                    question=question.question,
                    parsed_answer=parsed,
                    **parsed_spec.options,
                )
            )
        return bool(MATCHERS[parsed_spec.name](parsed, question.answer, **parsed_spec.options))
