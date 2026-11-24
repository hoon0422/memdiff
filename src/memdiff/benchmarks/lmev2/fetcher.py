"""Stream-fetch LongMemEval-V2-Small data without downloading 5.6 GB trajectory screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from tqdm import tqdm

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main"


def _download_file(client: httpx.Client, rel_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    url = f"{HF_BASE}/{rel_path}"
    with client.stream("GET", url, follow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fp:
            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                fp.write(chunk)


def fetch_lmev2_small(
    *,
    data_root: Path,
    domain: str = "enterprise",
    include_question_screenshots: bool = True,
) -> dict[str, int]:
    """Download questions, small haystack, optional 2.6MB question screenshots, and stream-filter trajectories."""
    data_root.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120.0) as client:
        for rel in ("SCHEMA.md", "DATA_CARD.md", "questions.jsonl", "haystacks/lme_v2_small.json"):
            _download_file(client, rel, data_root / rel)

        questions = [
            json.loads(line)
            for line in (data_root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        haystack = json.loads((data_root / "haystacks/lme_v2_small.json").read_text(encoding="utf-8"))

        selected_questions = [
            q for q in questions if domain == "all" or q.get("domain") == domain
        ]
        needed_traj_ids = {
            tid for q in selected_questions for tid in haystack[q["id"]]
        }

        downloaded_q_imgs = 0
        if include_question_screenshots:
            for q in tqdm(selected_questions, desc="Question screenshots", unit="q"):
                img_rel = q.get("image")
                if isinstance(img_rel, str) and img_rel.strip():
                    _download_file(client, img_rel, data_root / img_rel)
                    downloaded_q_imgs += 1

        out_traj_path = data_root / f"trajectories_small_{domain}.jsonl"
        kept = 0
        if not out_traj_path.exists():
            url = f"{HF_BASE}/trajectories.jsonl"
            with client.stream("GET", url, follow_redirects=True) as resp:
                resp.raise_for_status()
                total_bytes = int(resp.headers.get("content-length", 0))
                with (
                    out_traj_path.open("w", encoding="utf-8") as out_fp,
                    tqdm(
                        total=total_bytes or None,
                        unit="B",
                        unit_scale=True,
                        desc=f"Streaming trajectories ({domain}: {len(needed_traj_ids)} needed)",
                    ) as pbar,
                ):
                    buf = ""
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        pbar.update(len(chunk))
                        buf += chunk.decode("utf-8")
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            if not line.strip():
                                continue
                            # Fast ID prefix check before full json.loads
                            header_obj = json.loads(line)
                            if header_obj.get("id") in needed_traj_ids:
                                out_fp.write(line + "\n")
                                kept += 1
                    if buf.strip():
                        header_obj = json.loads(buf)
                        if header_obj.get("id") in needed_traj_ids:
                            out_fp.write(buf.strip() + "\n")
                            kept += 1
        else:
            kept = sum(1 for _ in out_traj_path.open("r", encoding="utf-8"))

        # Maintain a default symlink/copy alias `trajectories_small.jsonl`
        default_alias = data_root / "trajectories_small.jsonl"
        if default_alias.exists() or default_alias.is_symlink():
            default_alias.unlink()
        default_alias.symlink_to(out_traj_path.name)

    return {
        "questions_selected": len(selected_questions),
        "trajectories_needed": len(needed_traj_ids),
        "trajectories_saved": kept,
        "question_screenshots_checked": downloaded_q_imgs,
    }


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
    summary = fetch_lmev2_small(
        data_root=args.data_root,
        domain=args.domain,
        include_question_screenshots=args.include_question_screenshots,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
