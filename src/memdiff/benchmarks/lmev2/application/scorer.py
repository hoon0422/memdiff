"""Route an eval_function spec to a deterministic matcher or an LLM judge."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from memdiff.benchmarks.lmev2.application.judges import (
    ChatClient,
    llm_abstention_checker,
    llm_gotchas_checker,
)
from memdiff.benchmarks.lmev2.domain.scoring.matchers import MATCHERS
from memdiff.benchmarks.lmev2.domain.scoring.spec import EvalSpec


class ScoredQuestion(Protocol):
    question: str
    answer: str


JUDGES: dict[str, Callable[..., bool]] = {
    "llm_abstention_checker": llm_abstention_checker,
    "llm_gotchas_checker": llm_gotchas_checker,
}


class Scorer:
    def __init__(self, judge_client: ChatClient, judge_model: str | None) -> None:
        self._client = judge_client
        self._model = judge_model

    def score(self, spec: str, raw: str, parsed: str, question: ScoredQuestion) -> bool:
        """Judges grade the RAW response; deterministic matchers grade the PARSED answer."""
        parsed_spec = EvalSpec.parse(spec)
        if parsed_spec.is_judge:
            kwargs = {
                **parsed_spec.options,
                "question_text": question.question,
                "parsed_prediction": parsed,
                "llm_client": self._client,
                "judge_model": self._model,
            }
            return bool(JUDGES[parsed_spec.name](raw, question.answer, **kwargs))
        return bool(MATCHERS[parsed_spec.name](parsed, question.answer, **parsed_spec.options))
