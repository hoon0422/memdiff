"""Reader predictions on DriftSim queries, their deterministic scoring and aggregation."""

from __future__ import annotations

from pydantic import BaseModel

from memdiff.benchmarks.common.domain.answer import norm_phrase_set_match
from memdiff.benchmarks.drift_sim.domain.episode import DriftQuery


class DriftPrediction(BaseModel):
    """Field order is the predictions.jsonl key order."""

    query_id: str
    query_type: str
    entity: str
    question: str
    gold_answer: str
    stale_answers: list[str]
    raw_response: str
    parsed_answer: str
    is_correct: bool
    is_stale_error: bool
    retrieved_context: list[str]


class DriftMetrics(BaseModel):
    """Field order is the summary.json key order. Divisions are guarded to 0.0."""

    total_queries: int
    overall_accuracy: float
    dynamic_state_count: int
    dynamic_state_accuracy: float
    stale_belief_count: int
    stale_belief_rate: float
    stale_share_of_dynamic_errors: float
    durable_rule_count: int
    durable_rule_preservation: float


def score_drift_prediction(parsed_answer: str, query: DriftQuery) -> tuple[bool, bool]:
    """Return `(is_correct, is_stale_error)` using deterministic phrase matching."""
    is_correct = norm_phrase_set_match(parsed_answer, query.gold_answer)
    if is_correct:
        return True, False
    is_stale = any(
        norm_phrase_set_match(parsed_answer, stale_val) for stale_val in query.stale_answers
    )
    return False, is_stale


def aggregate_drift_predictions(predictions: list[DriftPrediction]) -> DriftMetrics:
    total = len(predictions)
    dyn = [p for p in predictions if p.query_type == "dynamic_state"]
    rules = [p for p in predictions if p.query_type == "durable_rule"]

    dyn_wrong = [p for p in dyn if not p.is_correct]
    stale_errors = [p for p in dyn if p.is_stale_error]

    return DriftMetrics(
        total_queries=total,
        overall_accuracy=sum(1 for p in predictions if p.is_correct) / total if total else 0.0,
        dynamic_state_count=len(dyn),
        dynamic_state_accuracy=sum(1 for p in dyn if p.is_correct) / len(dyn) if dyn else 0.0,
        stale_belief_count=len(stale_errors),
        stale_belief_rate=len(stale_errors) / len(dyn) if dyn else 0.0,
        stale_share_of_dynamic_errors=len(stale_errors) / len(dyn_wrong) if dyn_wrong else 0.0,
        durable_rule_count=len(rules),
        durable_rule_preservation=sum(1 for p in rules if p.is_correct) / len(rules)
        if rules
        else 0.0,
    )
