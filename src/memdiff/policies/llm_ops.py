"""LLM-backed belief extractor and revision reconciler for StructuredExtract & BeliefRevision policies."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from memdiff.core.models import Belief, BeliefType, Provenance, RevisionOp
from memdiff.llm.openai_client import CachedLLMClient
from memdiff.policies.belief_revision import RevisionDecision

_VALID_BELIEF_TYPES = {bt.value for bt in BeliefType}
_VALID_OPS = {
    RevisionOp.ADD.value,
    RevisionOp.SUPERSEDE.value,
    RevisionOp.CONSOLIDATE.value,
    RevisionOp.REFUTE.value,
    RevisionOp.IGNORE.value,
}

_EXTRACTOR_SYSTEM_PROMPT = """You extract structured, atomic memory beliefs from an agent trajectory or operational session.
Return ONLY valid JSON with key "beliefs" containing a list of objects:
{
  "beliefs": [
    {
      "type": "static_surface" | "dynamic_state" | "workflow" | "gotcha" | "durable_rule" | "temporary",
      "surface": "short scoped key like 'billing-api.deployment_region' or 'ServiceNow > Incidents > Filter'",
      "claim": "concise factual claim",
      "closed_set_items": ["optional exhaustive list of visible UI options/items if observed"],
      "confidence": 0.0 to 1.0,
      "durability": 0.0 to 1.0,
      "step_index": integer or null,
      "excerpt": "short grounding phrase from the log"
    }
  ]
}

Rules:
- Extract every state initialization/update (`dynamic_state`, durability ~0.85) and every durable policy/rule (`durable_rule`, durability ~0.95).
- For UI trajectories, extract reusable environment structure (`static_surface`), UI interaction transitions (`dynamic_state`), verified multi-step flows (`workflow`), and failure traps/workarounds (`gotcha`, especially when outcome=failure or a workaround succeeded).
- Mark one-off debug logs, ephemeral ticket/build/pod/PR IDs, or transient task noise as `temporary` with low durability (0.1 to 0.2).
- Keep claims concise, specific, and self-contained. Use consistent `surface` keys for the same entity/attribute or UI page.
- Output JSON only, no markdown commentary."""

_RECONCILER_SYSTEM_PROMPT = """You reconcile a NEW candidate belief against EXISTING active beliefs in memory.
Choose exactly ONE revision operation and return ONLY valid JSON:
{
  "op": "ADD" | "SUPERSEDE" | "CONSOLIDATE" | "REFUTE" | "IGNORE",
  "target_belief_id": "id of an existing belief from the list, or null for ADD/IGNORE",
  "merged_claim": "optional updated/merged claim for SUPERSEDE or CONSOLIDATE, else null",
  "reason": "short justification"
}

Decision rules:
- SUPERSEDE: The candidate updates or replaces an existing belief about the SAME entity/attribute/surface (e.g., new setting value replacing old value, or verified workaround replacing a failed attempt). Set target_belief_id to the outdated belief.
- CONSOLIDATE: The candidate confirms or adds complementary details/closed-set items to an existing belief without contradicting it. Set target_belief_id to that belief.
- REFUTE: The candidate proves an existing belief is false without providing a replacement claim.
- IGNORE: The candidate is redundant/identical to an existing active belief or is uninformative noise.
- ADD: The candidate is genuinely new and distinct from all listed active beliefs.
- Never supersede a `durable_rule` unless the candidate explicitly revokes that exact policy.
- Output JSON only."""


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _summarize_trajectory_for_extraction(
    trajectory: dict[str, Any],
    *,
    max_total_chars: int = 7000,
) -> str:
    tid = str(trajectory.get("id", "unknown"))
    domain = str(trajectory.get("domain", "unknown"))
    goal = str(trajectory.get("goal", ""))
    outcome = str(trajectory.get("outcome", "unknown"))
    states = trajectory.get("states", [])
    header = (
        f"Trajectory ID: {tid}\n"
        f"Domain: {domain}\n"
        f"Outcome: {outcome}\n"
        f"Goal: {goal[:300]}\n"
        f"Total states: {len(states) if isinstance(states, list) else 0}\n\n"
        "States:\n"
    )
    if not isinstance(states, list) or not states:
        return header + "(none)"

    # Fast full-fidelity path for concise sessions (e.g. DriftSim)
    if domain == "drift_sim":
        lines = [
            f"- [step {s.get('state_index', i)}] {s.get('accessibility_tree', s.get('text', ''))}"
            for i, s in enumerate(states)
            if isinstance(s, dict)
        ]
        return (header + "\n".join(lines))[:max_total_chars]

    # Sample/compress long UI trajectories (e.g., ServiceNow with 30+ states)
    budget = max(1000, max_total_chars - len(header))
    dict_states = [s for s in states if isinstance(s, dict)]
    max_states = 16
    if len(dict_states) > max_states:
        # Keep first 3, last 4, and evenly spaced middle states
        head = dict_states[:3]
        tail = dict_states[-4:]
        middle_pool = dict_states[3:-4]
        slots = max_states - len(head) - len(tail)
        stride = max(1, len(middle_pool) // max(1, slots))
        middle = middle_pool[::stride][:slots]
        sampled = head + middle + tail
    else:
        sampled = dict_states

    per_state_budget = max(180, budget // max(1, len(sampled)))
    parts: list[str] = []
    for s in sampled:
        sidx = s.get("state_index", s.get("step", 0))
        url = str(s.get("url", ""))
        action = s.get("action") or "<none>"
        thought = str(s.get("thought") or "")
        ax = str(s.get("accessibility_tree", s.get("text", "")))
        prefix = f"[step {sidx}] url={url} | action={action}"
        if thought:
            prefix += f" | thought={thought[:160]}"
        ax_budget = max(80, per_state_budget - len(prefix) - 4)
        ax_clean = re.sub(r"\s+", " ", ax).strip()[:ax_budget]
        parts.append(f"{prefix}\n  obs: {ax_clean}")

    return (header + "\n".join(parts))[:max_total_chars]


def _clamp_float(val: Any, default: float) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def make_llm_extractor(
    client: CachedLLMClient,
    *,
    model: str | None = None,
    think: bool | None = None,
    max_total_chars: int = 7000,
) -> Callable[[dict[str, Any], int], list[Belief]]:
    """Build an LLM-powered trajectory -> list[Belief] extractor compatible with Belief policies."""

    def extract(trajectory: dict[str, Any], step: int) -> list[Belief]:
        source_id = str(trajectory.get("id", f"traj_{step}"))
        outcome_raw = str(trajectory.get("outcome", "unknown")).lower()
        outcome: Any = outcome_raw if outcome_raw in {"success", "failure"} else "unknown"
        digest = _summarize_trajectory_for_extraction(trajectory, max_total_chars=max_total_chars)
        raw = client.chat(
            [
                {"role": "system", "content": _EXTRACTOR_SYSTEM_PROMPT},
                {"role": "user", "content": digest},
            ],
            model=model,
            temperature=0.0,
            max_completion_tokens=16384,
            think=think,
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return []
        raw_items = parsed.get("beliefs")
        if not isinstance(raw_items, list):
            return []

        beliefs: list[Belief] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            btype_str = str(item.get("type", "static_surface")).strip().lower()
            if btype_str not in _VALID_BELIEF_TYPES:
                btype_str = BeliefType.STATIC_SURFACE.value
            surface = str(item.get("surface", "general")).strip() or "general"
            closed_raw = item.get("closed_set_items")
            closed_items = (
                [str(x).strip() for x in closed_raw if str(x).strip()]
                if isinstance(closed_raw, list)
                else []
            )
            step_idx = item.get("step_index")
            step_idx_int = int(step_idx) if isinstance(step_idx, int) else None
            excerpt = str(item.get("excerpt", "")).strip() or None
            beliefs.append(
                Belief(
                    id=f"cand_{source_id}_{idx}",
                    type=BeliefType(btype_str),
                    surface=surface,
                    claim=claim,
                    closed_set_items=closed_items,
                    confidence=_clamp_float(item.get("confidence"), 0.8),
                    durability=_clamp_float(item.get("durability"), 0.8),
                    created_at_step=step,
                    updated_at_step=step,
                    provenance=[
                        Provenance(
                            source_id=source_id,
                            step_index=step_idx_int,
                            outcome=outcome,
                            excerpt=excerpt[:200] if excerpt else None,
                        )
                    ],
                )
            )
        return beliefs

    return extract


def make_llm_reconciler(
    client: CachedLLMClient,
    *,
    model: str | None = None,
    think: bool | None = None,
    min_neighbor_sim: float = 0.35,
) -> Callable[[Belief, list[tuple[Belief, float]]], RevisionDecision]:
    """Build an LLM-powered reconciler choosing ADD/SUPERSEDE/CONSOLIDATE/REFUTE/IGNORE."""

    def reconcile(
        candidate: Belief,
        neighbors: list[tuple[Belief, float]],
    ) -> RevisionDecision:
        relevant = [
            (b, sim)
            for b, sim in neighbors
            if sim >= min_neighbor_sim or b.surface.lower() == candidate.surface.lower()
        ]
        if not relevant:
            return RevisionDecision(RevisionOp.ADD, reason="no_similar_active_belief")

        valid_ids = {b.id for b, _ in relevant}
        neighbor_lines = []
        for b, sim in relevant:
            closed_str = f" | closed_set={b.closed_set_items}" if b.closed_set_items else ""
            neighbor_lines.append(
                f"- id={b.id} | sim={sim:.3f} | type={b.type.value} | surface={b.surface} | "
                f"step={b.updated_at_step} | claim={b.claim}{closed_str}"
            )

        user_prompt = (
            "NEW CANDIDATE BELIEF:\n"
            f"- type={candidate.type.value} | surface={candidate.surface} | "
            f"step={candidate.created_at_step} | claim={candidate.claim}"
            + (
                f" | closed_set={candidate.closed_set_items}"
                if candidate.closed_set_items
                else ""
            )
            + "\n\nEXISTING ACTIVE BELIEFS:\n"
            + "\n".join(neighbor_lines)
        )

        raw = client.chat(
            [
                {"role": "system", "content": _RECONCILER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.0,
            max_completion_tokens=16384,
            think=think,
        )
        parsed = _parse_json_object(raw)
        if not parsed:
            return RevisionDecision(RevisionOp.ADD, reason="reconciler_parse_fallback")

        op_str = str(parsed.get("op", "ADD")).strip().upper()
        if op_str not in _VALID_OPS:
            op_str = RevisionOp.ADD.value
        op = RevisionOp(op_str)

        target_id_raw = parsed.get("target_belief_id")
        target_id = str(target_id_raw).strip() if target_id_raw else None
        merged_raw = parsed.get("merged_claim")
        merged_claim = str(merged_raw).strip() if merged_raw else None
        reason = str(parsed.get("reason", "")).strip()

        if op in {RevisionOp.SUPERSEDE, RevisionOp.CONSOLIDATE, RevisionOp.REFUTE}:
            if not target_id or target_id not in valid_ids:
                return RevisionDecision(RevisionOp.ADD, reason="invalid_target_fallback")

        return RevisionDecision(
            op,
            target_belief_id=target_id,
            merged_claim=merged_claim,
            reason=reason,
        )

    return reconcile
