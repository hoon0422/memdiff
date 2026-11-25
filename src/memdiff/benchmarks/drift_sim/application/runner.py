"""Track 2 runner: ingest the episode into each policy's ledger, ask the reader, score, summarize."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from memdiff.audit.domain.replay import ReplayQuestionComparison
from memdiff.benchmarks.common.application.ports import UsageReport
from memdiff.benchmarks.common.application.workspace import (
    MemoryWorkspace,
    MemoryWorkspaceFactory,
    WorkspaceSpec,
)
from memdiff.benchmarks.common.domain.answer import extract_boxed_answer
from memdiff.benchmarks.drift_sim.application.ports import DriftReader, DriftRunStore
from memdiff.benchmarks.drift_sim.domain.episode import (
    DriftEpisode,
    DriftSimConfig,
    generate_episode,
)
from memdiff.benchmarks.drift_sim.domain.prediction import (
    DriftPrediction,
    aggregate_drift_predictions,
    score_drift_prediction,
)
from memdiff.memory.domain.trajectory import Trajectory


class DriftBenchmarkRunner:
    def __init__(
        self,
        *,
        workspaces: MemoryWorkspaceFactory,
        reader: DriftReader,
        runs: DriftRunStore,
        usage: UsageReport,
    ) -> None:
        self._workspaces = workspaces
        self._reader = reader
        self._runs = runs
        self._usage = usage

    def run(
        self,
        *,
        config: DriftSimConfig,
        policies: list[str],
        top_k: int = 6,
        recency_decay: float = 0.15,
        think: bool | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        episode = generate_episode(config)
        n_sessions = len(episode.sessions)

        def spec(name: str) -> WorkspaceSpec:
            return WorkspaceSpec(
                policy_name=name,
                db_path=self._runs.db_path(name),
                recency_decay=recency_decay,
                think=think,
            )

        results: dict[str, dict[str, Any]] = {}
        preds_by_policy: dict[str, list[DriftPrediction]] = {}
        for name in policies:
            # check before opening: opening a workspace creates the ledger file
            saved = self._runs.load_predictions(name) if resume else None
            if saved is not None and self._runs.db_exists(name):
                with self._workspaces.open(spec(name)) as ws:
                    ws.policy.resume_from(n_sessions)
                    metrics = aggregate_drift_predictions(saved).model_dump()
                    metrics |= {"policy_name": name, "top_k": top_k, "resumed": True}
                    metrics["memory_stats"] = ws.policy.stats().model_dump()
                results[name], preds_by_policy[name] = metrics, saved
                continue

            skip_ingest = resume and self._runs.db_exists(name)
            if not resume:
                self._runs.discard_db(name)
            with self._workspaces.open(spec(name)) as ws:
                metrics, preds = self._evaluate(
                    ws, episode, name, top_k=top_k, think=think, skip_ingest=skip_ingest
                )
            results[name], preds_by_policy[name] = metrics, preds
            self._runs.save_predictions(name, preds)

        self._export_replay(policies, preds_by_policy, spec, n_sessions=n_sessions)
        summary = {
            "benchmark": "drift_sim",
            "run_id": self._runs.run_id,
            "config": config.model_dump(),
            "top_k": top_k,
            "recency_decay": recency_decay,
            "llm_usage": self._usage.as_dict(),
            "policies": results,
        }
        self._runs.save_summary(summary)
        return summary

    def _evaluate(
        self,
        ws: MemoryWorkspace,
        episode: DriftEpisode,
        name: str,
        *,
        top_k: int,
        think: bool | None,
        skip_ingest: bool,
    ) -> tuple[dict[str, Any], list[DriftPrediction]]:
        t0 = time.perf_counter()
        if skip_ingest:
            ws.policy.resume_from(len(episode.sessions))
        else:
            for session in episode.sessions:
                ws.policy.insert(Trajectory.model_validate(session.to_trajectory_dict()))
        ingest_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        predictions: list[DriftPrediction] = []
        for query in episode.queries:
            ctx = ws.policy.query(query.question, top_k=top_k)
            texts = [c.value for c in ctx if c.type == "text"]
            raw = self._reader.answer(query.question, texts, think=think)
            parsed = extract_boxed_answer(raw)
            is_correct, is_stale = score_drift_prediction(parsed, query)
            predictions.append(
                DriftPrediction(
                    query_id=query.id,
                    query_type=query.query_type,
                    entity=query.entity,
                    question=query.question,
                    gold_answer=query.gold_answer,
                    stale_answers=query.stale_answers,
                    raw_response=raw,
                    parsed_answer=parsed,
                    is_correct=is_correct,
                    is_stale_error=is_stale,
                    retrieved_context=texts,
                )
            )
        query_seconds = time.perf_counter() - t1

        metrics = aggregate_drift_predictions(predictions).model_dump()
        metrics |= {
            "policy_name": name,
            "top_k": top_k,
            "ingest_seconds": round(ingest_seconds, 2),
            "query_seconds": round(query_seconds, 2),
            "memory_stats": ws.policy.stats().model_dump(),
        }
        return metrics, predictions

    def _export_replay(
        self,
        policies: list[str],
        preds_by_policy: dict[str, list[DriftPrediction]],
        spec: Callable[[str], WorkspaceSpec],
        *,
        n_sessions: int,
    ) -> None:
        """Replay bundle for belief_revision (or the last policy); it needs every policy's answers."""
        replay_policy = "belief_revision" if "belief_revision" in preds_by_policy else policies[-1]
        baseline = "naive_rag" if "naive_rag" in preds_by_policy else policies[0]
        comparisons: list[ReplayQuestionComparison] = []
        if replay_policy != baseline:
            for base_p, prop_p in zip(
                preds_by_policy[baseline], preds_by_policy[replay_policy], strict=True
            ):
                comparisons.append(
                    ReplayQuestionComparison(
                        question_id=base_p.query_id,
                        category=base_p.query_type,
                        question=base_p.question,
                        gold_answer=base_p.gold_answer,
                        baseline_policy=baseline,
                        baseline_answer=base_p.parsed_answer,
                        baseline_correct=base_p.is_correct,
                        baseline_retrieved=base_p.retrieved_context[:3],
                        proposed_policy=replay_policy,
                        proposed_answer=prop_p.parsed_answer,
                        proposed_correct=prop_p.is_correct,
                    )
                )
        with self._workspaces.open(spec(replay_policy)) as ws:  # reopen: the ledger is kept
            ws.policy.resume_from(n_sessions)
            bundle = ws.replay.build_bundle(
                run_id=self._runs.run_id,
                benchmark="drift_sim",
                policy_name=replay_policy,
                total_steps=n_sessions,
                stats=ws.policy.stats(),
                comparisons=comparisons,
            )
        self._runs.save_replay(replay_policy, bundle)
