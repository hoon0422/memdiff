"""LLM-backed `Embedder`, `BeliefExtractor` and `BeliefReconciler` over the `ChatModel` ports.

Request bytes must stay identical to the pre-refactor extractor and reconciler (pinned by the goldens in tests/golden).
"""

from __future__ import annotations

import json
import re
from typing import Any

from memdiff.llm.application.catalog import PromptCatalog
from memdiff.llm.application.ports import ChatModel, EmbeddingModel
from memdiff.llm.domain.messages import ChatMessage, ChatRequest
from memdiff.llm.domain.prompt import PromptId
from memdiff.memory.domain.belief import Belief, BeliefType, Provenance
from memdiff.memory.domain.revision import RevisionDecision, RevisionOp
from memdiff.memory.domain.trajectory import Trajectory

_VALID_BELIEF_TYPES = {bt.value for bt in BeliefType}
_VALID_OPS = {op.value for op in RevisionOp if op is not RevisionOp.FORGET}
_MAX_COMPLETION_TOKENS = 16384  # qwen3.5 thinks by default; keep room for `content`


class LLMEmbedder:
    def __init__(self, model: EmbeddingModel, *, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    def embed(self, texts: list[str]) -> list[Any]:
        return self._model.embed(texts, model=self._model_name).vectors


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
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_float(val: Any, default: float) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, f))


def summarize_trajectory(trajectory: Trajectory, *, max_total_chars: int = 7000) -> str:
    extra = trajectory.model_extra or {}
    domain = str(extra.get("domain", "unknown"))
    states = trajectory.states
    header = (
        f"Trajectory ID: {trajectory.id or 'unknown'}\n"
        f"Domain: {domain}\n"
        f"Outcome: {trajectory.outcome}\n"
        f"Goal: {trajectory.goal[:300]}\n"
        f"Total states: {len(states)}\n\n"
        "States:\n"
    )
    if not states:
        return header + "(none)"

    def state_index(i: int, s: Any) -> Any:
        if "state_index" in s.model_fields_set:
            return s.state_index
        return i if domain == "drift_sim" else (s.model_extra or {}).get("step", 0)

    # Concise sessions (DriftSim) are kept verbatim
    if domain == "drift_sim":
        lines = [f"- [step {state_index(i, s)}] {s.content}" for i, s in enumerate(states)]
        return (header + "\n".join(lines))[:max_total_chars]

    # Long UI trajectories: first 3, last 4, evenly spaced middle
    budget = max(1000, max_total_chars - len(header))
    indexed = list(enumerate(states))
    max_states = 16
    if len(indexed) > max_states:
        head, tail = indexed[:3], indexed[-4:]
        middle_pool = indexed[3:-4]
        slots = max_states - len(head) - len(tail)
        stride = max(1, len(middle_pool) // max(1, slots))
        sampled = head + middle_pool[::stride][:slots] + tail
    else:
        sampled = indexed

    per_state_budget = max(180, budget // max(1, len(sampled)))
    parts: list[str] = []
    for i, s in sampled:
        thought = str((s.model_extra or {}).get("thought") or "")
        prefix = f"[step {state_index(i, s)}] url={s.url} | action={s.action or '<none>'}"
        if thought:
            prefix += f" | thought={thought[:160]}"
        ax_budget = max(80, per_state_budget - len(prefix) - 4)
        ax_clean = re.sub(r"\s+", " ", s.content).strip()[:ax_budget]
        parts.append(f"{prefix}\n  obs: {ax_clean}")
    return (header + "\n".join(parts))[:max_total_chars]


def _ask(chat: ChatModel, system: str, user: str, *, model: str, think: bool | None) -> str:
    return chat.complete(
        [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        ChatRequest(
            model=model, temperature=0.0, max_tokens=_MAX_COMPLETION_TOKENS, think=think
        ),
    ).text


class LLMBeliefExtractor:
    def __init__(
        self,
        chat: ChatModel,
        prompts: PromptCatalog,
        *,
        model: str,
        think: bool | None = None,
        max_total_chars: int = 7000,
    ) -> None:
        self._chat, self._model, self._think = chat, model, think
        self._system = prompts.render(PromptId.EXTRACTOR_SYSTEM)
        self._max_total_chars = max_total_chars

    def extract(self, trajectory: Trajectory, step: int) -> list[Belief]:
        source_id = trajectory.id or f"traj_{step}"
        outcome: Any = trajectory.outcome.lower()
        if outcome not in {"success", "failure"}:
            outcome = "unknown"
        digest = summarize_trajectory(trajectory, max_total_chars=self._max_total_chars)
        raw = _ask(self._chat, self._system, digest, model=self._model, think=self._think)
        parsed = _parse_json_object(raw)
        raw_items = parsed.get("beliefs") if parsed else None
        if not isinstance(raw_items, list):
            return []

        beliefs: list[Belief] = []
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            btype = str(item.get("type", "static_surface")).strip().lower()
            if btype not in _VALID_BELIEF_TYPES:
                btype = BeliefType.STATIC_SURFACE.value
            closed_raw = item.get("closed_set_items")
            closed = (
                [str(x).strip() for x in closed_raw if str(x).strip()]
                if isinstance(closed_raw, list)
                else []
            )
            step_idx = item.get("step_index")
            excerpt = str(item.get("excerpt", "")).strip() or None
            beliefs.append(
                Belief(
                    id=f"cand_{source_id}_{idx}",
                    type=BeliefType(btype),
                    surface=str(item.get("surface", "general")).strip() or "general",
                    claim=claim,
                    closed_set_items=closed,
                    confidence=_clamp_float(item.get("confidence"), 0.8),
                    durability=_clamp_float(item.get("durability"), 0.8),
                    created_at_step=step,
                    updated_at_step=step,
                    provenance=[
                        Provenance(
                            source_id=source_id,
                            step_index=step_idx if isinstance(step_idx, int) else None,
                            outcome=outcome,
                            excerpt=excerpt[:200] if excerpt else None,
                        )
                    ],
                )
            )
        return beliefs


class LLMBeliefReconciler:
    def __init__(
        self,
        chat: ChatModel,
        prompts: PromptCatalog,
        *,
        model: str,
        think: bool | None = None,
        min_neighbor_sim: float = 0.35,
    ) -> None:
        self._chat, self._prompts, self._model, self._think = chat, prompts, model, think
        self._system = prompts.render(PromptId.RECONCILER_SYSTEM)
        self._min_sim = min_neighbor_sim

    def decide(
        self, candidate: Belief, neighbors: list[tuple[Belief, float]]
    ) -> RevisionDecision:
        relevant = [
            (b, sim)
            for b, sim in neighbors
            if sim >= self._min_sim or b.surface.lower() == candidate.surface.lower()
        ]
        if not relevant:
            return RevisionDecision(op=RevisionOp.ADD, reason="no_similar_active_belief")

        valid_ids = {b.id for b, _ in relevant}
        neighbor_lines = []
        for b, sim in relevant:
            closed = f" | closed_set={b.closed_set_items}" if b.closed_set_items else ""
            neighbor_lines.append(
                f"- id={b.id} | sim={sim:.3f} | type={b.type.value} | surface={b.surface} | "
                f"step={b.updated_at_step} | claim={b.claim}{closed}"
            )
        candidate_line = (
            f"- type={candidate.type.value} | surface={candidate.surface} | "
            f"step={candidate.created_at_step} | claim={candidate.claim}"
        )
        if candidate.closed_set_items:
            candidate_line += f" | closed_set={candidate.closed_set_items}"
        user = self._prompts.render(
            PromptId.RECONCILER_USER, candidate=candidate_line, neighbors="\n".join(neighbor_lines)
        )

        parsed = _parse_json_object(
            _ask(self._chat, self._system, user, model=self._model, think=self._think)
        )
        if not parsed:
            return RevisionDecision(op=RevisionOp.ADD, reason="reconciler_parse_fallback")

        op_str = str(parsed.get("op", "ADD")).strip().upper()
        op = RevisionOp(op_str if op_str in _VALID_OPS else RevisionOp.ADD.value)
        target_raw, merged_raw = parsed.get("target_belief_id"), parsed.get("merged_claim")
        target_id = str(target_raw).strip() if target_raw else None
        merged_claim = str(merged_raw).strip() if merged_raw else None
        if op in {RevisionOp.SUPERSEDE, RevisionOp.CONSOLIDATE, RevisionOp.REFUTE} and (
            not target_id or target_id not in valid_ids
        ):
            return RevisionDecision(op=RevisionOp.ADD, reason="invalid_target_fallback")
        return RevisionDecision(
            op=op,
            target_belief_id=target_id,
            merged_claim=merged_claim,
            reason=str(parsed.get("reason", "")).strip(),
        )
