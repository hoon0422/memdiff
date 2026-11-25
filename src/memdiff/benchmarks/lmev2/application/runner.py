"""LME-V2 benchmark runner: one policy per `run()`, one fresh memory workspace per haystack group."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from memdiff.benchmarks.common.application.workspace import MemoryWorkspaceFactory, WorkspaceSpec
from memdiff.benchmarks.common.domain.answer import extract_boxed_answer, is_unknown
from memdiff.benchmarks.lmev2.application.ports import LMEV2Dataset, LMEV2RunStore, Reader
from memdiff.benchmarks.lmev2.application.scorer import Scorer
from memdiff.benchmarks.lmev2.domain.dataset import (
    group_by_haystack,
    needed_trajectory_ids,
    select_questions,
)
from memdiff.benchmarks.lmev2.domain.scoring.aggregate import aggregate_metrics
from memdiff.benchmarks.lmev2.domain.scoring.categories import CATEGORY_MAP


class LMEV2RunParams(BaseModel):
    domain: str
    policy_name: str
    top_k: int = 6
    recency_decay: float = 0.05
    max_chunk_chars: int = 2400
    balanced_per_type: int | None = None
    limit: int | None = None
    include_images: bool = True
    think: bool | None = None


class LMEV2BenchmarkRunner:
    def __init__(
        self,
        *,
        dataset: LMEV2Dataset,
        workspaces: MemoryWorkspaceFactory,
        reader: Reader,
        scorer: Scorer,
        store: LMEV2RunStore,
        usage: Callable[[], dict[str, int]],
    ) -> None:
        self._dataset = dataset
        self._workspaces = workspaces
        self._reader = reader
        self._scorer = scorer
        self._store = store
        self._usage = usage

    def run(self, params: LMEV2RunParams) -> dict[str, Any]:
        policy = params.policy_name
        questions = select_questions(
            self._dataset.load_questions(domain=params.domain),
            balanced_per_type=params.balanced_per_type,
            limit=params.limit,
        )
        haystack = self._dataset.load_haystack()
        trajectories = self._dataset.load_trajectories(
            allowed_ids=needed_trajectory_ids(questions, haystack)
        )
        groups = group_by_haystack(questions, haystack)

        records: list[dict[str, Any]] = []
        ingest_seconds = 0.0
        query_seconds = 0.0
        last_stats: dict[str, Any] = {}

        for group_idx, (traj_order, group_qs) in enumerate(groups, start=1):
            spec = WorkspaceSpec(
                policy_name=policy,
                db_path=self._store.fresh_db_path(policy, group_idx),
                recency_decay=params.recency_decay,
                max_chunk_chars=params.max_chunk_chars,
                think=params.think,
            )
            with self._workspaces.open(spec) as ws:
                t0 = time.perf_counter()
                for tid in tqdm(
                    traj_order, desc=f"[{policy}] Ingest group {group_idx}/{len(groups)}", unit="traj"
                ):
                    if (traj := trajectories.get(tid)) is not None:
                        ws.policy.insert(traj)
                ingest_seconds += time.perf_counter() - t0
                last_stats = ws.policy.stats().model_dump()

                t1 = time.perf_counter()
                for q in tqdm(
                    group_qs, desc=f"[{policy}] Eval group {group_idx}/{len(groups)}", unit="q"
                ):
                    image = self._dataset.image_path(q) if params.include_images else None
                    ctx = ws.policy.query(
                        q.question, query_image=str(image) if image else None, top_k=params.top_k
                    )
                    raw = self._reader.answer(
                        domain=params.domain,
                        question=q,
                        context=ctx,
                        image_path=image,
                        think=params.think,
                    )
                    parsed = extract_boxed_answer(raw)
                    records.append(
                        {
                            "question_id": q.id,
                            "domain": q.domain,
                            "question_type": q.question_type,
                            "category": CATEGORY_MAP.get(q.question_type, q.question_type),
                            "is_abstention_problem": q.question_type.endswith("-abs"),
                            "eval_function": q.eval_function,
                            "question": q.question,
                            "gold_answer": q.answer,
                            "raw_response": raw,
                            "parsed_answer": parsed,
                            "is_unknown": is_unknown(parsed),
                            "score_bool": bool(self._scorer.score(q.eval_function, raw, parsed, q)),
                            "retrieved_count": len(ctx),
                        }
                    )
                query_seconds += time.perf_counter() - t1

                # known bug (#14): every group overwrites the previous group's replay
                self._store.write_replay(
                    policy,
                    ws.replay.build_bundle(
                        run_id=self._store.run_id,
                        benchmark=f"lmev2_small_{params.domain}",
                        policy_name=policy,
                        total_steps=len(traj_order),
                        stats=ws.policy.stats(),
                    ),
                )

        self._store.write_predictions(policy, records)
        summary = {
            "benchmark": f"lmev2_small_{params.domain}",
            "run_id": self._store.run_id,
            "policy_name": policy,
            "top_k": params.top_k,
            "max_chunk_chars": params.max_chunk_chars,
            "recency_decay": params.recency_decay,
            "balanced_per_type": params.balanced_per_type,
            "ingest_seconds": round(ingest_seconds, 2),
            "query_seconds": round(query_seconds, 2),
            "memory_stats": last_stats,
            "llm_usage": self._usage(),
            "metrics": aggregate_metrics(records),
        }
        self._store.write_summary(policy, summary)
        return summary
