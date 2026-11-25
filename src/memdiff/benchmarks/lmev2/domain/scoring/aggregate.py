"""Per-category accuracy aggregation."""

from __future__ import annotations

from typing import Any

from memdiff.benchmarks.lmev2.domain.scoring.categories import (
    ABSTENTION_CATEGORIES,
    NON_ABSTENTION_CATEGORIES,
)


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate accuracy, wrong-answer rate, and unknown rate by category."""
    if not records:
        return {"overall": {"count_all_questions": 0}}

    def _mean(rows: list[dict[str, Any]]) -> float | None:
        return sum(1.0 if r["score_bool"] else 0.0 for r in rows) / len(rows) if rows else None

    def _breakdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        if n == 0:
            return {
                "count": 0,
                "pct_correct": None,
                "pct_answered_wrong": None,
                "pct_unknown": None,
            }
        unknown_n = sum(1 for r in rows if r["is_unknown"])
        correct_n = sum(1 for r in rows if r["score_bool"] and not r["is_unknown"])
        wrong_n = n - correct_n - unknown_n
        return {
            "count": n,
            "pct_correct": correct_n / n,
            "pct_answered_wrong": wrong_n / n,
            "pct_unknown": unknown_n / n,
        }

    non_abst = [r for r in records if not r.get("is_abstention_problem")]
    abst = [r for r in records if r.get("is_abstention_problem")]
    return {
        "overall": {
            "overall_full_set": _mean(records),
            "overall_non_abstention_only": _mean(non_abst),
            "overall_abstention_only": _mean(abst),
            "count_all_questions": len(records),
            "count_non_abstention": len(non_abst),
            "count_abstention": len(abst),
        },
        "non_abstention_by_category": {
            cat: _breakdown([r for r in non_abst if r["category"] == cat])
            for cat in NON_ABSTENTION_CATEGORIES
        },
        "abstention_by_category": {
            cat: _breakdown([r for r in abst if r["category"] == cat])
            for cat in ABSTENTION_CATEGORIES
        },
    }
