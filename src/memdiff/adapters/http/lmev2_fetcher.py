"""Stream-fetch LongMemEval-V2-Small from HuggingFace without the 5.6 GB trajectory screenshots."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
from tqdm import tqdm

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/main"


class HuggingFaceLMEV2Fetcher:
    """`LMEV2Fetcher`. The `httpx.Client` is injected (tests pass a `MockTransport` client)."""

    def __init__(self, client: httpx.Client, *, base_url: str = HF_BASE) -> None:
        self._client = client
        self._base = base_url

    def _download_file(self, rel_path: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return
        with self._client.stream("GET", f"{self._base}/{rel_path}", follow_redirects=True) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fp:
                for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                    fp.write(chunk)

    def _stream_trajectories(self, dest: Path, needed_ids: set[str], domain: str) -> int:
        kept = 0
        with self._client.stream(
            "GET", f"{self._base}/trajectories.jsonl", follow_redirects=True
        ) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))
            with (
                dest.open("w", encoding="utf-8") as out_fp,
                tqdm(
                    total=total_bytes or None,
                    unit="B",
                    unit_scale=True,
                    desc=f"Streaming trajectories ({domain}: {len(needed_ids)} needed)",
                ) as pbar,
            ):

                def keep(line: str) -> bool:
                    if not line.strip():
                        return False
                    if json.loads(line).get("id") in needed_ids:
                        out_fp.write(line + "\n")
                        return True
                    return False

                buf = ""
                for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                    pbar.update(len(chunk))
                    buf += chunk.decode("utf-8")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        kept += keep(line)
                kept += keep(buf.strip())
        return kept

    def fetch(
        self, *, data_root: Path, domain: str, include_question_screenshots: bool
    ) -> dict[str, int]:
        data_root.mkdir(parents=True, exist_ok=True)
        for rel in ("SCHEMA.md", "DATA_CARD.md", "questions.jsonl", "haystacks/lme_v2_small.json"):
            self._download_file(rel, data_root / rel)

        questions = [
            json.loads(line)
            for line in (data_root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        haystack = json.loads((data_root / "haystacks/lme_v2_small.json").read_text(encoding="utf-8"))
        selected = [q for q in questions if domain == "all" or q.get("domain") == domain]
        needed_ids = {tid for q in selected for tid in haystack[q["id"]]}

        checked_imgs = 0
        if include_question_screenshots:
            for q in tqdm(selected, desc="Question screenshots", unit="q"):
                img_rel = q.get("image")
                if isinstance(img_rel, str) and img_rel.strip():
                    self._download_file(img_rel, data_root / img_rel)
                    checked_imgs += 1

        out_traj_path = data_root / f"trajectories_small_{domain}.jsonl"
        if out_traj_path.exists():
            with out_traj_path.open("r", encoding="utf-8") as fp:
                kept = sum(1 for _ in fp)
        else:
            kept = self._stream_trajectories(out_traj_path, needed_ids, domain)

        alias = data_root / "trajectories_small.jsonl"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(out_traj_path.name)

        return {
            "questions_selected": len(selected),
            "trajectories_needed": len(needed_ids),
            "trajectories_saved": kept,
            "question_screenshots_checked": checked_imgs,
        }
