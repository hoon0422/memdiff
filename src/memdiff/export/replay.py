from __future__ import annotations

from pathlib import Path

from memdiff.core.models import ReplayBundle, ReplayQuestionComparison
from memdiff.core.store import BeliefStore


def build_replay_bundle(
    *,
    store: BeliefStore,
    run_id: str,
    benchmark: str,
    policy_name: str,
    total_steps: int,
    comparisons: list[ReplayQuestionComparison] | None = None,
) -> ReplayBundle:
    return ReplayBundle(
        run_id=run_id,
        benchmark=benchmark,
        policy_name=policy_name,
        total_steps=total_steps,
        beliefs=store.list_beliefs(),
        edges=store.list_edges(),
        events=store.list_events(),
        stats=store.stats(total_ingested_sources=total_steps),
        comparisons=list(comparisons or []),
    )


def export_replay_json(bundle: ReplayBundle, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
