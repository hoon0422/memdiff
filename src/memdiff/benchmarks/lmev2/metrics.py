"""Compatibility facade: scoring now lives in `common.domain.answer` and `lmev2.{domain,application}`.

Kept until #12 removes it; new code imports from the real modules.
"""

from __future__ import annotations

from typing import Any

from memdiff.benchmarks.common.domain.answer import (
    DEFAULT_SEPARATORS,
    extract_boxed_answer,
    is_unknown,
    norm_phrase_set_match,
    normalize_phrase,
    split_phrases,
)
from memdiff.benchmarks.lmev2.application.judges import llm_abstention_checker, llm_gotchas_checker
from memdiff.benchmarks.lmev2.application.scorer import JUDGES
from memdiff.benchmarks.lmev2.domain.scoring.aggregate import aggregate_metrics
from memdiff.benchmarks.lmev2.domain.scoring.categories import (
    ABSTENTION_CATEGORIES,
    CATEGORY_MAP,
    NON_ABSTENTION_CATEGORIES,
)
from memdiff.benchmarks.lmev2.domain.scoring.matchers import (
    MATCHERS,
    mc_choice_match,
    mc_choice_set_match,
    norm_phrase_set_match_ordered,
)
from memdiff.benchmarks.lmev2.domain.scoring.spec import EvalSpec


def eval_from_spec(spec: str, prediction: str | None, answer: str | None, **overrides: Any) -> bool:
    parsed = EvalSpec.parse(spec)
    fn = {**MATCHERS, **JUDGES}[parsed.name]
    return bool(fn(prediction, answer, **{**parsed.options, **overrides}))


__all__ = [
    "ABSTENTION_CATEGORIES",
    "CATEGORY_MAP",
    "DEFAULT_SEPARATORS",
    "NON_ABSTENTION_CATEGORIES",
    "aggregate_metrics",
    "eval_from_spec",
    "extract_boxed_answer",
    "is_unknown",
    "llm_abstention_checker",
    "llm_gotchas_checker",
    "mc_choice_match",
    "mc_choice_set_match",
    "norm_phrase_set_match",
    "norm_phrase_set_match_ordered",
    "normalize_phrase",
    "split_phrases",
]
