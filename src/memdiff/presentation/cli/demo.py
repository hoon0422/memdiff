"""Hands-on demo: feed a few DriftSim sessions into BeliefRevisionPolicy, query it, inspect the ledger.

Offline by default: a rule-based extractor/reconciler and hashed bag-of-words embeddings stand in
for the LLM, so it runs in seconds with no Ollama. `--live` uses the configured LLM instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dishka import Provider

from memdiff.adapters.replay.json_file import JsonFileReplayWriter
from memdiff.benchmarks.common.application.workspace import MemoryWorkspaceFactory, WorkspaceSpec
from memdiff.benchmarks.drift_sim.simulator import DriftSimConfig, generate_episode
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.providers import OfflineProvider
from memdiff.bootstrap.settings import get_settings
from memdiff.memory.domain.belief import BeliefStatus
from memdiff.memory.domain.trajectory import Trajectory


def run_demo(
    *, out_dir: Path, sessions: int, top_k: int, live: bool = False, think: bool | None = None
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "demo.sqlite"
    db_path.unlink(missing_ok=True)
    episode = generate_episode(
        DriftSimConfig(
            num_sessions=sessions, num_entities=3, num_durable_rules=2, noise_events_per_session=1
        )
    )
    overrides: list[Provider] = [] if live else [OfflineProvider()]
    # the rule-based reconciler matches by surface, so the offline run looks wide
    spec = WorkspaceSpec(
        policy_name="belief_revision",
        db_path=db_path,
        think=think,
        **({} if live else {"candidate_top_k": 20}),
    )

    container = build_container(get_settings(), *overrides)
    try:
        with container.get(MemoryWorkspaceFactory).open(spec) as ws:
            print("=== 1) insert(): one session per step ===")
            for session in episode.sessions:
                print(f"\n--- step {session.step} ({session.session_id}) input events:")
                for ev in session.events:
                    print(f"    {ev}")
                ws.policy.insert(Trajectory.model_validate(session.to_trajectory_dict()))
                print("  ledger ops:")
                for ev in ws.log.list_events():
                    if ev.step == session.step:
                        print(f"    [{ev.op.value:<11}] {ev.summary}")

            print("\n=== 2) active beliefs after all sessions ===")
            for b in ws.beliefs.list_beliefs(status=BeliefStatus.ACTIVE):
                print(f"  {b.id} [{b.type.value}] {b.surface}: {b.claim}")

            print(f"\n=== 3) query() top_k={top_k} for each ground-truth question ===")
            hits = 0
            for q in episode.queries:
                ctx = ws.policy.query(q.question, top_k=top_k)
                top = ctx[0].value if ctx else "(nothing)"
                hit = q.gold_answer.lower() in top.lower()
                hits += hit
                print(
                    f"\n  Q: {q.question}\n  gold={q.gold_answer!r} stale={q.stale_answers}"
                    f" -> top-1 {'HIT' if hit else 'MISS'}"
                )
                print("    " + top.replace("\n", "\n    "))

            stats = ws.policy.stats()
            replay_path = out_dir / "replay_demo.json"
            JsonFileReplayWriter(replay_path).write(
                ws.replay.build_bundle(
                    run_id="demo",
                    benchmark="drift_sim_demo",
                    policy_name="belief_revision",
                    total_steps=len(episode.sessions),
                    stats=stats,
                )
            )
    finally:
        container.close()

    print("\n=== 4) stats ===")
    print(f"  {stats.model_dump()}")
    print(f"  top-1 hits: {hits}/{len(episode.queries)}")
    print(
        f"\nSQLite ledger: {db_path}   "
        f"(try: sqlite3 {db_path} 'SELECT step, op, summary FROM timeline_events')"
    )
    print(f"Replay JSON:   {replay_path}   (load it in the web viewer: pnpm --dir web dev)")
    return {"hits": hits, "queries": len(episode.queries), **stats.revision_counts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try the MemDiff belief ledger on a tiny DriftSim episode."
    )
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    parser.add_argument(
        "--live", action="store_true", help="Use the configured Ollama/OpenAI LLM + embeddings."
    )
    parser.add_argument("--no-think", action="store_true", help="With --live: disable qwen thinking.")
    args = parser.parse_args()
    run_demo(
        out_dir=args.out_dir,
        sessions=args.sessions,
        top_k=args.top_k,
        live=args.live,
        think=False if args.no_think else None,
    )
