"""Golden-snapshot helpers shared by the characterization tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

GOLDEN_ROOT = Path(__file__).parent / "golden"
UPDATE_ENV = "MEMDIFF_UPDATE_GOLDEN"


def assert_golden(relpath: str, actual: str) -> None:
    """Compare `actual` to `tests/golden/<relpath>`; rewrite it only when MEMDIFF_UPDATE_GOLDEN=1."""
    path = GOLDEN_ROOT / relpath
    if os.environ.get(UPDATE_ENV) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    assert path.exists(), f"Missing golden {path}; generate it with {UPDATE_ENV}=1"
    assert actual == path.read_text(encoding="utf-8"), (
        f"{relpath} differs from golden; if the change is intended, rerun with {UPDATE_ENV}=1"
    )


def zero_timings(obj: Any) -> Any:
    """Wall-clock fields (`*_seconds`) are the only nondeterministic summary values."""
    if isinstance(obj, dict):
        return {
            k: 0.0 if k.endswith("_seconds") else zero_timings(v) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [zero_timings(v) for v in obj]
    return obj


def dump_json(obj: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=sort_keys) + "\n"


def assert_run_matches_golden(
    prefix: str,
    *,
    requests: list[dict[str, Any]],
    out_dir: Path,
    summary_file: str,
    files: dict[str, str],
) -> None:
    """Snapshot the request log, the summary/metrics JSON, and the other run artifacts.

    `files` maps golden file name -> artifact file name inside `out_dir`.
    """
    assert_golden(f"{prefix}/requests.json", dump_json(requests, sort_keys=True))
    summary = json.loads((out_dir / summary_file).read_text(encoding="utf-8"))
    assert_golden(f"{prefix}/summary.json", dump_json(zero_timings(summary)))
    for golden_name, artifact in files.items():
        assert_golden(f"{prefix}/{golden_name}", (out_dir / artifact).read_text(encoding="utf-8"))
