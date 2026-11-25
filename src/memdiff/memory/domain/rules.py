"""Pure memory rules: noise filtering, forgetting, and consolidation arithmetic."""

from __future__ import annotations

from .belief import Belief, BeliefType, Provenance

CONSOLIDATE_CONFIDENCE_BOOST = 0.05
FORGETTABLE_MAX_DURABILITY = 0.5


def is_noise(candidate: Belief, *, threshold: float) -> bool:
    """Ephemeral task noise: a temporary candidate below the durability threshold."""
    return candidate.type == BeliefType.TEMPORARY and candidate.durability < threshold


def should_forget(belief: Belief, *, current_step: int, max_age: int) -> bool:
    return (
        belief.type == BeliefType.TEMPORARY
        and current_step - belief.updated_at_step >= max_age
        and belief.durability <= FORGETTABLE_MAX_DURABILITY
    )


def consolidate(
    belief: Belief,
    *,
    step: int,
    provenance: Provenance,
    merged_claim: str | None,
    closed_set_items: list[str],
) -> Belief:
    closed = (
        sorted(set(belief.closed_set_items) | set(closed_set_items))
        if closed_set_items
        else belief.closed_set_items
    )
    return belief.model_copy(
        update={
            "claim": merged_claim or belief.claim,
            "closed_set_items": closed,
            "confidence": min(1.0, belief.confidence + CONSOLIDATE_CONFIDENCE_BOOST),
            "evidence_count": belief.evidence_count + 1,
            "updated_at_step": step,
            "provenance": [*belief.provenance, provenance],
        }
    )
