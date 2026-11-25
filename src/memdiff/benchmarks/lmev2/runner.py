"""Evaluation runner for Track 1 (LongMemEval-V2 Small)."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from memdiff.benchmarks.common.domain.answer import extract_boxed_answer, is_unknown
from memdiff.benchmarks.lmev2.application.scorer import Scorer
from memdiff.benchmarks.lmev2.domain.scoring.aggregate import aggregate_metrics
from memdiff.benchmarks.lmev2.domain.scoring.categories import CATEGORY_MAP
from memdiff.benchmarks.lmev2.loader import (
    LMEV2Question,
    load_haystack,
    load_questions,
    load_trajectories,
)
from memdiff.config import Settings, get_settings
from memdiff.core.models import MemoryContextItem
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

DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    "web": (
        "You are an experienced colleague in a web browsing environment that has "
        "a customized magento-based shopping website, a customized magento-based "
        "shopping admin cms website, as well as a customized forum website based "
        "on reddit/postmill. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed."
    ),
    "enterprise": (
        "You are an experienced colleague working in a customized ServiceNow "
        "environment. Answer based on your memory of the environment. "
        "If you do not know the answer, output exactly \\boxed{UNKNOWN}. "
        "Do not guess. Never attempt to guess an answer if you are not sure. "
        "If you believe the question's construction/premise is wrong, provide an "
        "explanation in \\boxed{} explaining why the question is flawed."
    ),
}


def select_balanced_questions(
    questions: list[LMEV2Question],
    *,
    per_type: int,
) -> list[LMEV2Question]:
    """Deterministically pick up to `per_type` questions from each question_type."""
    buckets: dict[str, list[LMEV2Question]] = defaultdict(list)
    for q in questions:
        buckets[q.question_type].append(q)
    selected: list[LMEV2Question] = []
    for qtype in sorted(buckets.keys()):
        selected.extend(buckets[qtype][:per_type])
    return selected


def _image_to_data_url(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_reader_messages(
    *,
    domain: str,
    question: LMEV2Question,
    context_items: list[MemoryContextItem],
    data_root: Path,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DOMAIN_SYSTEM_PROMPTS["enterprise"])
    text_chunks = [c["value"] for c in context_items if c.get("type") == "text"]
    memory_block = "\n\n---\n\n".join(text_chunks) if text_chunks else "(no retrieved memory)"

    user_text = (
        f"Retrieved Memory:\n{memory_block}\n\n"
        f"Question:\n{question.question}\n\n"
        "Answer based strictly on your memory above, and place your final answer inside \\boxed{...}."
    )

    if include_images and question.image:
        img_path = data_root / question.image
        if img_path.exists():
            return [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(img_path)},
                        },
                    ],
                },
            ]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def build_default_policy(
    policy_name: str,
    *,
    store: BeliefStore,
    client: CachedLLMClient,
    recency_decay: float = 0.05,
    max_chunk_chars: int = 2400,
    think: bool | None = None,
) -> MemoryPolicy:
    if policy_name == "naive_rag":
        return NaiveRagPolicy(
            store=store, embed_fn=client.embed, max_chunk_chars=max_chunk_chars
        )
    if policy_name == "recency_rag":
        return RecencyRagPolicy(
            store=store,
            embed_fn=client.embed,
            recency_decay=recency_decay,
            max_chunk_chars=max_chunk_chars,
        )
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


def run_lmev2_policy(
    *,
    data_root: Path,
    domain: str,
    policy_name: str,
    output_dir: Path,
    top_k: int = 6,
    recency_decay: float = 0.05,
    max_chunk_chars: int = 2400,
    balanced_per_type: int | None = None,
    limit: int | None = None,
    include_images: bool = True,
    think: bool | None = None,
    settings: Settings | None = None,
    client: CachedLLMClient | None = None,
    policy_factory: Callable[[BeliefStore], MemoryPolicy] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = settings or get_settings()
    owns_client = client is None
    llm = client or CachedLLMClient(cfg)
    scorer = Scorer(llm, judge_model=cfg.memdiff_judge_model)

    questions = load_questions(data_root, domain=domain)
    if balanced_per_type is not None:
        questions = select_balanced_questions(questions, per_type=balanced_per_type)
    elif limit is not None:
        questions = questions[:limit]

    haystack = load_haystack(data_root, tier="small")
    needed_ids = {tid for q in questions for tid in haystack[q.id]}
    trajectories = load_trajectories(data_root, allowed_ids=needed_ids)

    # Group questions by shared ordered haystack so we ingest each haystack once
    groups: dict[tuple[str, ...], list[LMEV2Question]] = defaultdict(list)
    for q in questions:
        groups[tuple(haystack[q.id])].append(q)

    records: list[dict[str, Any]] = []
    ingest_seconds = 0.0
    query_seconds = 0.0
    last_stats: dict[str, Any] = {}

    try:
        for group_idx, (traj_order, group_qs) in enumerate(groups.items(), start=1):
            db_path = output_dir / f"{policy_name}_group{group_idx}.sqlite"
            if db_path.exists():
                db_path.unlink()
            store = BeliefStore(db_path)
            try:
                policy = (
                    policy_factory(store)
                    if policy_factory is not None
                    else build_default_policy(
                        policy_name,
                        store=store,
                        client=llm,
                        recency_decay=recency_decay,
                        max_chunk_chars=max_chunk_chars,
                        think=think,
                    )
                )

                t0 = time.perf_counter()
                for tid in tqdm(
                    traj_order,
                    desc=f"[{policy_name}] Ingest group {group_idx}/{len(groups)}",
                    unit="traj",
                ):
                    traj = trajectories.get(tid)
                    if traj is not None:
                        policy.insert(traj)
                ingest_seconds += time.perf_counter() - t0
                last_stats = policy.stats().model_dump()

                t1 = time.perf_counter()
                for q in tqdm(
                    group_qs,
                    desc=f"[{policy_name}] Eval group {group_idx}/{len(groups)}",
                    unit="q",
                ):
                    q_img_path = (
                        str(data_root / q.image)
                        if include_images and q.image and (data_root / q.image).exists()
                        else None
                    )
                    ctx = policy.query(q.question, query_image=q_img_path, top_k=top_k)
                    messages = _build_reader_messages(
                        domain=domain,
                        question=q,
                        context_items=ctx,
                        data_root=data_root,
                        include_images=include_images,
                    )
                    raw_resp = llm.chat(
                        messages,
                        model=cfg.memdiff_chat_model,
                        temperature=0.0,
                        max_completion_tokens=16384,
                        think=think,
                    )
                    parsed = extract_boxed_answer(raw_resp)
                    unknown = is_unknown(parsed)
                    is_abst = q.question_type.endswith("-abs")
                    category = CATEGORY_MAP.get(q.question_type, q.question_type)

                    score_bool = scorer.score(q.eval_function, raw_resp, parsed, q)

                    records.append(
                        {
                            "question_id": q.id,
                            "domain": q.domain,
                            "question_type": q.question_type,
                            "category": category,
                            "is_abstention_problem": is_abst,
                            "eval_function": q.eval_function,
                            "question": q.question,
                            "gold_answer": q.answer,
                            "raw_response": raw_resp,
                            "parsed_answer": parsed,
                            "is_unknown": unknown,
                            "score_bool": bool(score_bool),
                            "retrieved_count": len(ctx),
                        }
                    )
                query_seconds += time.perf_counter() - t1

                bundle = build_replay_bundle(
                    store=store,
                    run_id=output_dir.name,
                    benchmark=f"lmev2_small_{domain}",
                    policy_name=policy_name,
                    total_steps=len(traj_order),
                )
                export_replay_json(bundle, output_dir / f"replay_{policy_name}.json")
            finally:
                store.close()

        pred_path = output_dir / f"{policy_name}_predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as fp:
            for rec in records:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

        aggregated = aggregate_metrics(records)
        summary = {
            "benchmark": f"lmev2_small_{domain}",
            "run_id": output_dir.name,
            "policy_name": policy_name,
            "top_k": top_k,
            "max_chunk_chars": max_chunk_chars,
            "recency_decay": recency_decay,
            "balanced_per_type": balanced_per_type,
            "ingest_seconds": round(ingest_seconds, 2),
            "query_seconds": round(query_seconds, 2),
            "memory_stats": last_stats,
            "llm_usage": llm.usage.as_dict(),
            "metrics": aggregated,
        }
        (output_dir / f"{policy_name}_metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        if owns_client:
            llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track 1 (LongMemEval-V2 Small) evaluation.")
    parser.add_argument("--data-root", type=Path, default=Path("data/longmemeval-v2"))
    parser.add_argument("--domain", choices=["enterprise", "web"], default="enterprise")
    parser.add_argument("--policy", default="naive_rag")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--recency-decay", type=float, default=0.05)
    parser.add_argument("--max-chunk-chars", type=int, default=2400)
    parser.add_argument("--balanced-per-type", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-images", action="store_true", default=False)
    parser.add_argument("--no-think", action="store_true", default=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    run_id = args.run_id or f"lmev2_{args.domain}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    summary = run_lmev2_policy(
        data_root=args.data_root,
        domain=args.domain,
        policy_name=args.policy,
        output_dir=args.runs_dir / run_id,
        top_k=args.top_k,
        recency_decay=args.recency_decay,
        max_chunk_chars=args.max_chunk_chars,
        balanced_per_type=args.balanced_per_type,
        limit=args.limit,
        include_images=not args.no_images,
        think=False if args.no_think else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
