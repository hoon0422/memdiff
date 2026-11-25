"""Rule-based stand-ins for the LLM embedder/extractor/reconciler (offline demo, no Ollama)."""

from __future__ import annotations

import re
import zlib

import numpy as np

from memdiff.memory.domain.belief import Belief, BeliefType, Provenance
from memdiff.memory.domain.revision import RevisionDecision, RevisionOp
from memdiff.memory.domain.trajectory import Trajectory

_DIM = 256
_INIT = re.compile(r"\[STATE INIT\] (\S+) (\S+) is set to (.+)\.$")
_UPDATE = re.compile(r"\[STATE UPDATE\] (\S+) (\S+) changed from .+ to (.+?) \(")
_RULE = re.compile(r"\[DURABLE POLICY:(\w+)\] (.+)")


class OfflineEmbedder:
    """Hashed bag-of-words vectors (crc32 keeps them stable across runs, unlike hash())."""

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out = []
        for text in texts:
            vec = np.zeros(_DIM, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9_.\-]+", text.lower()):
                vec[zlib.crc32(tok.encode()) % _DIM] += 1.0
            out.append(vec / (np.linalg.norm(vec) or 1.0))
        return out


class OfflineBeliefExtractor:
    """Parses DriftSim event lines into candidate beliefs (what the LLM extractor does live)."""

    def extract(self, trajectory: Trajectory, step: int) -> list[Belief]:
        sid = str(trajectory.id)
        beliefs: list[Belief] = []
        for state in trajectory.states:
            text = state.accessibility_tree or ""
            if m := _INIT.match(text) or _UPDATE.match(text):
                svc, attr, val = m.groups()
                btype, surface = BeliefType.DYNAMIC_STATE, f"{svc}.{attr}"
                claim, dur = f"{svc} {attr} is {val}", 0.85
            elif m := _RULE.match(text):
                btype, surface = BeliefType.DURABLE_RULE, m[1]
                claim, dur = f"Policy {m[1]}: {m[2]}", 0.95
            else:
                btype, surface, claim, dur = BeliefType.TEMPORARY, "ops.noise", text, 0.1
            beliefs.append(
                Belief(
                    id=f"cand_{sid}_{len(beliefs)}",
                    type=btype,
                    surface=surface,
                    claim=claim,
                    durability=dur,
                    created_at_step=step,
                    updated_at_step=step,
                    provenance=[
                        Provenance(
                            source_id=sid,
                            step_index=state.state_index,
                            outcome="success",
                            excerpt=text[:200],
                        )
                    ],
                )
            )
        return beliefs


class OfflineBeliefReconciler:
    """Same surface + same claim -> IGNORE, same surface + new claim -> SUPERSEDE, else ADD."""

    def decide(
        self, candidate: Belief, neighbors: list[tuple[Belief, float]]
    ) -> RevisionDecision:
        for old, _sim in neighbors:
            if old.surface == candidate.surface:
                if old.claim == candidate.claim:
                    return RevisionDecision(op=RevisionOp.IGNORE, reason=f"duplicate of {old.id}")
                return RevisionDecision(
                    op=RevisionOp.SUPERSEDE,
                    target_belief_id=old.id,
                    reason="newer value for same surface",
                )
        return RevisionDecision(op=RevisionOp.ADD, reason="new surface")
