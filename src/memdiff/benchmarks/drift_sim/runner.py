"""Evaluation runner for Track 2 (DriftSim controlled belief-revision benchmark)."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memdiff.benchmarks.drift_sim.simulator import (
    DriftEpisode,
    DriftQuery,
    DriftSimConfig,
    generate_episode,
)
from memdiff.benchmarks.lmev2.metrics import extract_boxed_answer, norm_phrase_set_match
from memdiff.config import Settings, get_settings
from memdiff.core.models import MemoryContextItem, ReplayQuestionComparison
from memdiff.core.protocol import MemoryPolicy
from memdiff.core.store import BeliefStore
from memdiff.export.replay import build_replay_bundle, export_replay_json
from memdiff.llm.openai_client import CachedLLMClient
from memdiff.policies import (
    BeliefRevisionPolicy,
    NaiveRagPolicy,
    RecencyRagPolicy,
    StructuredExtractPolicy,
)
from memdiff.policies.llm_ops import make_llm_extractor, make_llm_reconciler

DRIFT_SYSTEM_PROMPT = (
    "You are an operations assistant answering questions strictly from retrieved memory notes. "
    "Sessions are ordered chronologically (higher session numbers are newer). "
    "Always answer with the most current valid value or the exact required durable policy threshold. "
    "Put ONLY the concise final answer inside \\boxed{...}. "
    "If the memory does not contain the answer, output \\boxed{UNKNOWN}."
)


@dataclass
class DriftPrediction:
    query_id: str
    query_type: str
    entity: str
    question: str
    gold_answer: str
    stale_answers: list[str]
    raw_response: str
    parsed_answer: str
    is_correct: bool
    is_stale_error: bool
    retrieved_context: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "entity": self.entity,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "stale_answers": self.stale_answers,
            "raw_response": self.raw_response,
            "parsed_answer": self.parsed_answer,
            "is_correct": self.is_correct,
            "is_stale_error": self.is_stale_error,
            "retrieved_context": self.retrieved_context,
        }


def score_drift_prediction(parsed_answer: str, query: DriftQuery) -> tuple[bool, bool]:
    """Return `(is_correct, is_stale_error)` using deterministic phrase matching."""
    is_correct = norm_phrase_set_match(parsed_answer, query.gold_answer)
    if is_correct:
        return True, False
    is_stale = any(
        norm_phrase_set_match(parsed_answer, stale_val) for stale_val in query.stale_answers
    )
    return False, is_stale


def _format_context_block(items: list[MemoryContextItem]) -> str:
    texts = [item["value"] for item in items if item.get("type") == "text"]
    if not texts:
        return "(no retrieved memory)"
    return "\n\n---\n\n".join(texts)


def _answer_drift_query(
    client: CachedLLMClient,
    query: DriftQuery,
    context_items: list[MemoryContextItem],
    *,
    model: str | None = None,
    think: bool | None = None,
) -> str:
    context_block = _format_context_block(context_items)
    user_prompt = (
        f"Retrieved Memory:\n{context_block}\n\n"
        f"Question:\n{query.question}\n\n"
        "Respond briefly and put your final answer inside \\boxed{...}."
    )
    return client.chat(
        [
            {"role": "system", "content": DRIFT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.0,
        max_completion_tokens=16384,
        think=think,
    )


def aggregate_drift_predictions(predictions: list[DriftPrediction]) -> dict[str, Any]:
    total = len(predictions)
    dyn = [p for p in predictions if p.query_type == "dynamic_state"]
    rules = [p for p in predictions if p.query_type == "durable_rule"]

    dyn_wrong = [p for p in dyn if not p.is_correct]
    stale_errors = [p for p in dyn if p.is_stale_error]

    return {
        "total_queries": total,
        "overall_accuracy": sum(1 for p in predictions if p.is_correct) / total if total else 0.0,
        "dynamic_state_count": len(dyn),
        "dynamic_state_accuracy": sum(1 for p in dyn if p.is_correct) / len(dyn) if dyn else 0.0,
        "stale_belief_count": len(stale_errors),
        "stale_belief_rate": len(stale_errors) / len(dyn) if dyn else 0.0,
        "stale_share_of_dynamic_errors": (
            len(stale_errors) / len(dyn_wrong) if dyn_wrong else 0.0
        ),
        "durable_rule_count": len(rules),
        "durable_rule_preservation": (
            sum(1 for p in rules if p.is_correct) / len(rules) if rules else 0.0
        ),
    }


def build_default_policy(
    policy_name: str,
    *,
    store: BeliefStore,
    client: CachedLLMClient,
    recency_decay: float = 0.15,
    think: bool | None = None,
) -> MemoryPolicy:
    if policy_name == "naive_rag":
        return NaiveRagPolicy(store=store, embed_fn=client.embed)
    if policy_name == "recency_rag":
        return RecencyRagPolicy(store=store, embed_fn=client.embed, recency_decay=recency_decay)
    if policy_name == "structured_extract":
        return StructuredExtractPolicy(
            store=store,
            embed_fn=client.embed,
            extractor_fn=make_llm_extractor(client, think=think),
        )
    if policy_name == "belief_revision":
        return BeliefRevisionPolicy(
            store=store,
            embed_fn=client.embed,
            extractor_fn=make_llm_extractor(client, think=think),
            reconciler_fn=make_llm_reconciler(client, think=think),
        )
    raise ValueError(
        f"Unknown policy '{policy_name}'. Expected one of: "
        "naive_rag, recency_rag, structured_extract, belief_revision."
    )


def evaluate_policy_on_episode(
    *,
    episode: DriftEpisode,
    policy_name: str,
    policy_factory: Callable[[BeliefStore], MemoryPolicy],
    client: CachedLLMClient,
    top_k: int = 6,
    reader_model: str | None = None,
    think: bool | None = None,
    db_path: str | Path = ":memory:",
    skip_ingest: bool = False,
) -> tuple[dict[str, Any], list[DriftPrediction], BeliefStore]:
    store = BeliefStore(db_path)
    policy = policy_factory(store)

    t0 = time.perf_counter()
    if skip_ingest:
        if hasattr(policy, "_step"):
            policy._step = len(episode.sessions)
    else:
        for session in episode.sessions:
            policy.insert(session.to_trajectory_dict())
    ingest_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    predictions: list[DriftPrediction] = []
    for query in episode.queries:
        ctx = policy.query(query.question, top_k=top_k)
        raw = _answer_drift_query(client, query, ctx, model=reader_model, think=think)
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
                retrieved_context=[c["value"] for c in ctx if c.get("type") == "text"],
            )
        )
    query_seconds = time.perf_counter() - t1

    metrics = aggregate_drift_predictions(predictions)
    metrics["policy_name"] = policy_name
    metrics["top_k"] = top_k
    metrics["ingest_seconds"] = round(ingest_seconds, 2)
    metrics["query_seconds"] = round(query_seconds, 2)
    metrics["memory_stats"] = policy.stats().model_dump()
    return metrics, predictions, store


def run_drift_sim_benchmark(
    *,
    config: DriftSimConfig,
    policies: list[str],
    output_dir: Path,
    top_k: int = 6,
    recency_decay: float = 0.15,
    think: bool | None = None,
    resume: bool = False,
    settings: Settings | None = None,
    client: CachedLLMClient | None = None,
    policy_factories: dict[str, Callable[[BeliefStore], MemoryPolicy]] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = generate_episode(config)
    owns_client = client is None
    llm = client or CachedLLMClient(settings or get_settings())

    results_by_policy: dict[str, dict[str, Any]] = {}
    preds_by_policy: dict[str, list[DriftPrediction]] = {}
    stores: dict[str, BeliefStore] = {}

    try:
        for name in policies:
            factory = (
                policy_factories[name]
                if policy_factories and name in policy_factories
                else (
                    lambda s, _name=name: build_default_policy(
                        _name,
                        store=s,
                        client=llm,
                        recency_decay=recency_decay,
                        think=think,
                    )
                )
            )
            db_path = output_dir / f"{name}.sqlite"
            pred_path = output_dir / f"{name}_predictions.jsonl"

            if resume and db_path.exists() and pred_path.exists():
                store = BeliefStore(db_path)
                preds = [
                    DriftPrediction(**json.loads(line))
                    for line in pred_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                metrics = aggregate_drift_predictions(preds)
                metrics["policy_name"] = name
                metrics["top_k"] = top_k
                metrics["resumed"] = True
                metrics["memory_stats"] = store.stats(
                    total_ingested_sources=len(episode.sessions)
                ).model_dump()
                results_by_policy[name] = metrics
                preds_by_policy[name] = preds
                stores[name] = store
                continue

            skip_ingest = False
            if resume and db_path.exists():
                skip_ingest = True
            elif db_path.exists():
                db_path.unlink()

            metrics, preds, store = evaluate_policy_on_episode(
                episode=episode,
                policy_name=name,
                policy_factory=factory,
                client=llm,
                top_k=top_k,
                think=think,
                db_path=db_path,
                skip_ingest=skip_ingest,
            )
            results_by_policy[name] = metrics
            preds_by_policy[name] = preds
            stores[name] = store

            with pred_path.open("w", encoding="utf-8") as fp:
                for pred in preds:
                    fp.write(json.dumps(pred.as_dict(), ensure_ascii=False) + "\n")

        # Export replay bundle for belief_revision (or the last policy if not present)
        replay_policy = "belief_revision" if "belief_revision" in stores else policies[-1]
        baseline_policy = "naive_rag" if "naive_rag" in preds_by_policy else policies[0]
        comparisons: list[ReplayQuestionComparison] = []
        if replay_policy != baseline_policy:
            for base_p, prop_p in zip(
                preds_by_policy[baseline_policy],
                preds_by_policy[replay_policy],
                strict=True,
            ):
                comparisons.append(
                    ReplayQuestionComparison(
                        question_id=base_p.query_id,
                        category=base_p.query_type,
                        question=base_p.question,
                        gold_answer=base_p.gold_answer,
                        baseline_policy=baseline_policy,
                        baseline_answer=base_p.parsed_answer,
                        baseline_correct=base_p.is_correct,
                        baseline_retrieved=base_p.retrieved_context[:3],
                        proposed_policy=replay_policy,
                        proposed_answer=prop_p.parsed_answer,
                        proposed_correct=prop_p.is_correct,
                    )
                )

        bundle = build_replay_bundle(
            store=stores[replay_policy],
            run_id=output_dir.name,
            benchmark="drift_sim",
            policy_name=replay_policy,
            total_steps=len(episode.sessions),
            comparisons=comparisons,
        )
        export_replay_json(bundle, output_dir / f"replay_{replay_policy}.json")

        summary = {
            "benchmark": "drift_sim",
            "run_id": output_dir.name,
            "config": config.model_dump(),
            "top_k": top_k,
            "recency_decay": recency_decay,
            "llm_usage": llm.usage.as_dict(),
            "policies": results_by_policy,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        for store in stores.values():
            store.close()
        if owns_client:
            llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track 2 (DriftSim) memory evaluation.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["naive_rag", "recency_rag"],
        help="Policies to evaluate (default: naive_rag recency_rag).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=24)
    parser.add_argument("--entities", type=int, default=8)
    parser.add_argument("--contradiction-rate", type=float, default=0.45)
    parser.add_argument("--noise-per-session", type=int, default=2)
    parser.add_argument("--durable-rules", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--recency-decay", type=float, default=0.15)
    parser.add_argument("--no-think", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    run_id = args.run_id or f"drift_sim_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    cfg = DriftSimConfig(
        seed=args.seed,
        num_sessions=args.sessions,
        num_entities=args.entities,
        contradiction_rate=args.contradiction_rate,
        noise_events_per_session=args.noise_per_session,
        num_durable_rules=args.durable_rules,
    )
    summary = run_drift_sim_benchmark(
        config=cfg,
        policies=args.policies,
        output_dir=args.runs_dir / run_id,
        top_k=args.top_k,
        recency_decay=args.recency_decay,
        think=False if args.no_think else None,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
