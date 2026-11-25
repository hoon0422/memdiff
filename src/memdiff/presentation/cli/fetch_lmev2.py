"""CLI to stream-fetch LongMemEval-V2-Small text data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memdiff.benchmarks.lmev2.application.ports import LMEV2Fetcher
from memdiff.bootstrap.container import build_container
from memdiff.bootstrap.lmev2_provider import LMEV2FetchProvider
from memdiff.bootstrap.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch LongMemEval-V2-Small text data (skips 5.6 GB trajectory screenshots)."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/longmemeval-v2"))
    parser.add_argument("--domain", choices=["enterprise", "web", "all"], default="enterprise")
    parser.add_argument(
        "--include-question-screenshots",
        action="store_true",
        default=True,
        help="Download the ~2.6MB question_screenshots/*.png for errors-gotchas evaluation.",
    )
    parser.add_argument(
        "--no-question-screenshots",
        dest="include_question_screenshots",
        action="store_false",
    )
    args = parser.parse_args()

    container = build_container(get_settings(), LMEV2FetchProvider())
    try:
        summary = container.get(LMEV2Fetcher).fetch(
            data_root=args.data_root,
            domain=args.domain,
            include_question_screenshots=args.include_question_screenshots,
        )
    finally:
        container.close()
    print(json.dumps(summary, indent=2))
