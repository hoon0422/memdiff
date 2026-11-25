#!/usr/bin/env python3
"""Hands-on demo: feed a few DriftSim sessions into BeliefRevisionPolicy, query it, inspect the ledger.

Offline by default: a rule-based extractor/reconciler and hashed bag-of-words embeddings stand in
for the LLM, so it runs in seconds with no Ollama. `--live` swaps in the real LLM callbacks.
"""

from __future__ import annotations

import argparse
import re
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from memdiff.benchmarks.drift_sim import DriftSimConfig, generate_episode
from memdiff.core.models import Belief, BeliefStatus, BeliefType, Provenance, RevisionOp
from memdiff.core.protocol import MemoryPolicy
from memdiff.core.store import BeliefStore
from memdiff.export.replay import build_replay_bundle, export_replay_json
from memdiff.policies import BeliefRevisionPolicy
from memdiff.policies.belief_revision import RevisionDecision

_DIM = 256
_INIT = re.compile(r"\[STATE INIT\] (\S+) (\S+) is set to (.+)\.$")
_UPDATE = re.compile(r"\[STATE UPDATE\] (\S+) (\S+) changed from .+ to (.+?) \(")
_RULE = re.compile(r"\[DURABLE POLICY:(\w+)\] (.+)")


def offline_embed(texts: list[str]) -> list[np.ndarray]:
    """Hashed bag-of-words vectors (crc32 keeps them stable across runs, unlike hash())."""
    out = []
    for text in texts:
        vec = np.zeros(_DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9_.\-]+", text.lower()):
            vec[zlib.crc32(tok.encode()) % _DIM] += 1.0
        out.append(vec / (np.linalg.norm(vec) or 1.0))
    return out


def offline_extract(trajectory: dict[str, Any], step: int) -> list[Belief]:
    """Parse DriftSim event lines into candidate beliefs (what the LLM extractor does in --live)."""
    sid = str(trajectory["id"])
    beliefs: list[Belief] = []
    for state in trajectory.get("states", []):
        text = str(state.get("accessibility_tree", ""))
        if m := _INIT.match(text) or _UPDATE.match(text):
            svc, attr, val = m.groups()
            btype, surface, claim, dur = BeliefType.DYNAMIC_STATE, f"{svc}.{attr}", f"{svc} {attr} is {val}", 0.85
        elif m := _RULE.match(text):
            btype, surface, claim, dur = BeliefType.DURABLE_RULE, m[1], f"Policy {m[1]}: {m[2]}", 0.95
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
                        step_index=state.get("state_index"),
                        outcome="success",
                        excerpt=text[:200],
                    )
                ],
            )
        )
    return beliefs


def offline_reconcile(cand: Belief, neighbors: list[tuple[Belief, float]]) -> RevisionDecision:
    """Same surface + same claim -> IGNORE, same surface + new claim -> SUPERSEDE, else ADD."""
    for old, _sim in neighbors:
        if old.surface == cand.surface:
            if old.claim == cand.claim:
                return RevisionDecision(RevisionOp.IGNORE, reason=f"duplicate of {old.id}")
            return RevisionDecision(
                RevisionOp.SUPERSEDE, target_belief_id=old.id, reason="newer value for same surface"
            )
    return RevisionDecision(RevisionOp.ADD, reason="new surface")


def build_policy(store: BeliefStore, *, live: bool, think: bool | None) -> MemoryPolicy:
    if live:
        from memdiff.benchmarks.drift_sim.runner import build_default_policy
        from memdiff.llm import CachedLLMClient

        return build_default_policy("belief_revision", store=store, client=CachedLLMClient(), think=think)
    return BeliefRevisionPolicy(
        store=store,
        embed_fn=offline_embed,
        extractor_fn=offline_extract,
        reconciler_fn=offline_reconcile,
        candidate_top_k=20,  # rule-based reconciler matches by surface, so look wide
    )


def run_demo(
    *, out_dir: Path, sessions: int, top_k: int, live: bool = False, think: bool | None = None
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "demo.sqlite"
    db_path.unlink(missing_ok=True)
    episode = generate_episode(
        DriftSimConfig(num_sessions=sessions, num_entities=3, num_durable_rules=2, noise_events_per_session=1)
    )

    with BeliefStore(db_path) as store:
        policy = build_policy(store, live=live, think=think)

        print("=== 1) insert(): one session per step ===")
        for session in episode.sessions:
            print(f"\n--- step {session.step} ({session.session_id}) input events:")
            for ev in session.events:
                print(f"    {ev}")
            policy.insert(session.to_trajectory_dict())
            print("  ledger ops:")
            for ev in store.list_events():
                if ev.step == session.step:
                    print(f"    [{ev.op.value:<11}] {ev.summary}")

        print("\n=== 2) active beliefs after all sessions ===")
        for b in store.list_beliefs(status=BeliefStatus.ACTIVE):
            print(f"  {b.id} [{b.type.value}] {b.surface}: {b.claim}")

        print(f"\n=== 3) query() top_k={top_k} for each ground-truth question ===")
        hits = 0
        for q in episode.queries:
            ctx = policy.query(q.question, top_k=top_k)
            top = ctx[0]["value"] if ctx else "(nothing)"
            hit = q.gold_answer.lower() in top.lower()
            hits += hit
            print(f"\n  Q: {q.question}\n  gold={q.gold_answer!r} stale={q.stale_answers} -> top-1 {'HIT' if hit else 'MISS'}")
            print("    " + top.replace("\n", "\n    "))

        stats = policy.stats()
        replay = export_replay_json(
            build_replay_bundle(
                store=store,
                run_id="demo",
                benchmark="drift_sim_demo",
                policy_name="belief_revision",
                total_steps=len(episode.sessions),
            ),
            out_dir / "replay_demo.json",
        )

    print("\n=== 4) stats ===")
    print(f"  {stats.model_dump()}")
    print(f"  top-1 hits: {hits}/{len(episode.queries)}")
    print(f"\nSQLite ledger: {db_path}   (try: sqlite3 {db_path} 'SELECT step, op, summary FROM timeline_events')")
    print(f"Replay JSON:   {replay}   (load it in the web viewer: pnpm --dir web dev)")
    return {"hits": hits, "queries": len(episode.queries), **stats.revision_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Try the MemDiff belief ledger on a tiny DriftSim episode.")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument("--live", action="store_true", help="Use the configured Ollama/OpenAI LLM + embeddings.")
    parser.add_argument("--no-think", action="store_true", help="With --live: disable qwen thinking.")
    args = parser.parse_args()
    run_demo(
        out_dir=args.out_dir,
        sessions=args.sessions,
        top_k=args.top_k,
        live=args.live,
        think=False if args.no_think else None,
    )


if __name__ == "__main__":
    main()
