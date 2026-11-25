"""CLI for Track 2 (DriftSim): argparse -> container -> `DriftBenchmarkRunner`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from dishka import Provider

from memdiff.benchmarks.drift_sim.application.runner import DriftBenchmarkRunner
from memdiff.benchmarks.drift_sim.domain.episode import DriftSimConfig
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.drift_providers import DriftSimProvider
from memdiff.bootstrap.settings import Settings, get_settings


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    run(argv, settings=get_settings())


def run(argv: Sequence[str] | None, *, settings: Settings, overrides: Sequence[Provider] = ()) -> None:
    """`main()` minus the `.env` lookup; tests pass their own settings and override providers."""
    args = build_parser().parse_args(argv)
    run_id = args.run_id or f"drift_sim_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    config = DriftSimConfig(
        seed=args.seed,
        num_sessions=args.sessions,
        num_entities=args.entities,
        contradiction_rate=args.contradiction_rate,
        noise_events_per_session=args.noise_per_session,
        num_durable_rules=args.durable_rules,
    )
    container = build_container(settings, DriftSimProvider(args.runs_dir), *overrides)
    try:
        summary = container.get(DriftBenchmarkRunner).run(
            config=config,
            policies=args.policies,
            run_id=run_id,
            top_k=args.top_k,
            recency_decay=args.recency_decay,
            think=False if args.no_think else None,
            resume=args.resume,
        )
    finally:
        container.close()
    print(json.dumps(summary, indent=2))
