from __future__ import annotations

from memdiff.benchmarks.lmev2.metrics import (
    aggregate_metrics,
    eval_from_spec,
    extract_boxed_answer,
    is_unknown,
)


def test_extract_boxed_and_unknown() -> None:
    assert extract_boxed_answer("Final answer is \\boxed{300}.") == "300"
    assert extract_boxed_answer("Reasoning... \\boxed{UNKNOWN}") == "UNKNOWN"
    assert is_unknown(extract_boxed_answer("\\boxed{unknown}"))


def test_deterministic_eval_specs() -> None:
    assert eval_from_spec(
        "norm_phrase_set_match|lower=true|normalize_hyphen=true|strip_punct=true|separators=,;|require_non_empty=true",
        "My Open Incidents, Incident Mobile, and Incident Portal",
        "Incident Mobile, Incident Portal, My Open Incidents",
    )
    assert eval_from_spec(
        "norm_phrase_set_match_ordered|lower=true|normalize_hyphen=true|strip_punct=true|separators=;|require_non_empty=true",
        "First open Reports, then click Problems",
        "Reports;Problems",
    )
    assert not eval_from_spec(
        "norm_phrase_set_match_ordered|lower=true|normalize_hyphen=true|strip_punct=true|separators=;|require_non_empty=true",
        "First open Problems, then Reports",
        "Reports;Problems",
    )
    assert eval_from_spec("mc_choice_match|require_non_empty=true", "Option C.", "C")
    assert eval_from_spec("mc_choice_set_match|require_non_empty=true", "Choices A and C", "CA")


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
