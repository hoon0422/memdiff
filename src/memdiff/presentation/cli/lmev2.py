"""CLI for Track 1 (LongMemEval-V2 Small): argparse -> container -> runner."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from memdiff.benchmarks.lmev2.application.runner import LMEV2BenchmarkRunner, LMEV2RunParams
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.lmev2_provider import LMEV2Provider
from memdiff.bootstrap.settings import get_settings


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
    container = build_container(
        get_settings(),
        LMEV2Provider(data_root=args.data_root, output_dir=args.runs_dir / run_id),
    )
    try:
        summary = container.get(LMEV2BenchmarkRunner).run(
            LMEV2RunParams(
                domain=args.domain,
                policy_name=args.policy,
                top_k=args.top_k,
                recency_decay=args.recency_decay,
                max_chunk_chars=args.max_chunk_chars,
                balanced_per_type=args.balanced_per_type,
                limit=args.limit,
                include_images=not args.no_images,
                think=False if args.no_think else None,
            )
        )
    finally:
        container.close()
    print(json.dumps(summary, indent=2))
