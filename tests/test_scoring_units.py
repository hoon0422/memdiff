from __future__ import annotations

import pytest

from memdiff.benchmarks.common.domain.answer import (
    extract_boxed_answer,
    is_unknown,
    norm_phrase_set_match,
    normalize_phrase,
    split_phrases,
)
from memdiff.benchmarks.lmev2.domain.scoring.aggregate import aggregate_metrics
from memdiff.benchmarks.lmev2.domain.scoring.judgement import parse_binary_judgement
from memdiff.benchmarks.lmev2.domain.scoring.matchers import (
    mc_choice_match,
    mc_choice_set_match,
    norm_phrase_set_match_ordered,
)
from memdiff.benchmarks.lmev2.domain.scoring.spec import EvalSpec


def test_answer_helpers() -> None:
    assert extract_boxed_answer("\\boxed{a} then \\boxed{b{c}}") == "b{c}"
    assert extract_boxed_answer("  plain ") == "plain"
    assert extract_boxed_answer("\\boxed{ }") == "\\boxed{ }"
    assert is_unknown(" UnKnown ")
    assert normalize_phrase("Open-Incident_X!") == "open incident x"
    assert split_phrases("5% canary, 15 minutes") == ["5 canary", "15 minutes"]
    assert norm_phrase_set_match("b and a", "a, b")


def test_ordered_and_mc_matchers() -> None:
    assert norm_phrase_set_match_ordered("a then b", "a;b", separators=[";"])
    assert not norm_phrase_set_match_ordered("b then a", "a;b", separators=[";"])
    assert mc_choice_match("\\boxed{Option c.}", "C")
    assert mc_choice_set_match("Final answers: A and C", "CA")
    assert not mc_choice_set_match("A", "AB")


def test_spec_parse_options() -> None:
    spec = EvalSpec.parse(
        "norm_phrase_set_match| lower=TRUE |separators=,; |strip_chars=.|x=none|y=[1]"
    )
    assert spec.name == "norm_phrase_set_match"
    assert spec.options == {
        "lower": True,
        "separators": [",", ";"],
        "strip_chars": ".",
        "x": None,
        "y": "[1]",
    }
    assert EvalSpec.parse('mc_choice_match|separators=[";", ","]').options["separators"] == [
        ";",
        ",",
    ]
    assert EvalSpec.parse("mc_choice_match|separator=").options["separator"] == []
    assert EvalSpec.parse("llm_gotchas_checker").is_judge
    assert not spec.is_judge


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "Empty eval_function spec"),
        (" | ", "Empty eval_function spec"),
        ("nope", "Unknown eval function: nope"),
    ],
)
def test_spec_parse_errors(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EvalSpec.parse(raw)


def test_spec_missing_equals_raises() -> None:
    with pytest.raises(ValueError):
        EvalSpec.parse("mc_choice_match|require_non_empty")


def test_parse_binary_judgement() -> None:
    assert parse_binary_judgement('```json\n{"label": "1", "reason": "ok"}\n```') == (1, "ok")
    assert parse_binary_judgement('{"label": 0}') == (0, "")
    assert parse_binary_judgement("label=0 because") == (0, "label=0 because")
    assert parse_binary_judgement('{"label": 5} "label": 1')[0] == 1
    with pytest.raises(ValueError, match="Could not parse"):
        parse_binary_judgement("no verdict")


def test_aggregate_empty() -> None:
    assert aggregate_metrics([]) == {"overall": {"count_all_questions": 0}}


def test_aggregate_metrics_breakdown() -> None:
    records = [
        {"category": "static", "is_abstention_problem": False, "score_bool": True, "is_unknown": False},
        {"category": "static", "is_abstention_problem": False, "score_bool": False, "is_unknown": True},
        {"category": "dynamic-abs", "is_abstention_problem": True, "score_bool": True, "is_unknown": False},
    ]
    agg = aggregate_metrics(records)
    assert agg["overall"]["count_all_questions"] == 3
    assert agg["non_abstention_by_category"]["static"]["pct_correct"] == 0.5
    assert agg["non_abstention_by_category"]["static"]["pct_unknown"] == 0.5
    assert agg["abstention_by_category"]["dynamic-abs"]["pct_correct"] == 1.0
