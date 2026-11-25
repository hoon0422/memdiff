from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from memdiff.memory.domain.belief import Belief, BeliefStatus, BeliefType, Provenance
from memdiff.memory.domain.rules import consolidate, is_noise, should_forget


def _belief(**kw: object) -> Belief:
    base: dict[str, object] = dict(
        id="b1", type=BeliefType.DYNAMIC_STATE, surface="s", claim="c", step=1, source_id="src"
    )
    base.update(kw)
    return Belief.new(**base)  # type: ignore[arg-type]


def test_domain_models_are_pydantic_and_validate_ranges() -> None:
    assert issubclass(Belief, BaseModel) and issubclass(Provenance, BaseModel)
    with pytest.raises(ValidationError):
        _belief(confidence=1.5)
    with pytest.raises(ValidationError):
        _belief(durability=-0.1)


def test_new_belief_defaults_provenance_to_source() -> None:
    b = _belief()
    assert b.provenance == [Provenance(source_id="src")]
    assert (b.status, b.evidence_count, b.created_at_step, b.updated_at_step) == (
        BeliefStatus.ACTIVE, 1, 1, 1,
    )


def test_status_transitions_return_new_rows_and_never_mutate() -> None:
    b = _belief()
    sup = b.mark_superseded(by="b2", step=2)
    assert (sup.status, sup.superseded_by, sup.updated_at_step) == (BeliefStatus.SUPERSEDED, "b2", 2)
    assert b.status == BeliefStatus.ACTIVE
    assert b.mark_refuted(step=3).status == BeliefStatus.REFUTED
    assert b.mark_forgotten(step=3).status == BeliefStatus.FORGOTTEN


def test_consolidate_adds_evidence_unions_closed_set_and_caps_confidence() -> None:
    b = _belief(confidence=0.98, closed_set_items=["b", "a"])
    out = consolidate(
        b, step=4, provenance=Provenance(source_id="s2"), merged_claim=None, closed_set_items=["c", "a"]
    )
    assert out.evidence_count == 2 and len(out.provenance) == 2
    assert out.confidence == 1.0
    assert out.closed_set_items == ["a", "b", "c"]
    assert out.claim == "c" and out.updated_at_step == 4
    keep = consolidate(b, step=4, provenance=Provenance(source_id="s2"), merged_claim="m", closed_set_items=[])
    assert keep.claim == "m" and keep.closed_set_items == ["b", "a"]


def test_noise_and_forgetting_rules() -> None:
    tmp = _belief(type=BeliefType.TEMPORARY, durability=0.1)
    assert is_noise(tmp, threshold=0.25)
    assert not is_noise(_belief(type=BeliefType.TEMPORARY, durability=0.25), threshold=0.25)
    assert not is_noise(_belief(durability=0.1), threshold=0.25)
    old = _belief(type=BeliefType.TEMPORARY, durability=0.5)
    assert should_forget(old, current_step=6, max_age=5)
    assert not should_forget(old, current_step=5, max_age=5)
    assert not should_forget(_belief(type=BeliefType.TEMPORARY, durability=0.6), current_step=9, max_age=5)
