"""Fake OpenAI-compatible SDK for characterization tests.

Plugs into `CachedLLMClient(client=...)`: records every chat/embedding request in order and
answers deterministically per role (extractor / reconciler / reader / judge), keyed off the
request content. Roles are detected from the system prompt, so a changed prompt that no longer
matches a role fails loudly instead of silently getting a canned answer.
"""

from __future__ import annotations

import ast
import json
import re
import zlib
from types import SimpleNamespace
from typing import Any

EMBED_DIMS = 32

_ROLES = (
    ("extractor", "You extract structured, atomic memory beliefs"),
    ("reconciler", "You reconcile a NEW candidate belief"),
    ("drift_reader", "You are an operations assistant"),
    ("lmev2_reader", "You are an experienced colleague"),
    ("judge_abstention", "You are a strict grader for flawed-premise"),
    ("judge_gotchas", "You are a strict grader for gotchas-style"),
)

_TAG = re.compile(
    r"\[(STATE INIT|STATE UPDATE|TEMP LOG|TEMP|DURABLE POLICY:(?P<rule>\w+)|"
    r"SURFACE (?P<surface>[^\]]+)|ERROR (?P<error>[^\]]+)|RETRACT (?P<retract>[^\]]+))\]\s*(?P<text>.+)"
)
_STEP = re.compile(r"\[step (\d+)\]")


def _embed(text: str) -> list[float]:
    vec = [0.0] * EMBED_DIMS
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[zlib.crc32(word.encode()) % EMBED_DIMS] += 1.0
    return vec


def _text_of(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(p["text"] for p in content if p.get("type") == "text")
    return str(content)


def _extract(user: str) -> str:
    outcome = re.search(r"^Outcome: (\w+)", user, flags=re.MULTILINE)
    failed = bool(outcome and outcome.group(1) == "failure")
    beliefs: list[dict[str, Any]] = []
    step = 0
    for line in user.splitlines():
        if (m_step := _STEP.search(line)) is not None:
            step = int(m_step.group(1))
        m = _TAG.search(line)
        if m is None:
            continue
        text, kind = m.group("text").strip(), m.group(1)
        item: dict[str, Any] = {"closed_set_items": [], "step_index": step, "excerpt": text[:40]}
        if kind in {"STATE INIT", "STATE UPDATE"}:
            surface = ".".join(text.split()[:2])
            item |= {"type": "dynamic_state", "surface": surface, "claim": text,
                     "confidence": 0.9, "durability": 0.85}
        elif m.group("rule"):
            item |= {"type": "durable_rule", "surface": m.group("rule"),
                     "claim": f"[DURABLE POLICY:{m.group('rule')}] {text}",
                     "confidence": 0.95, "durability": 0.95}
        elif kind.startswith("TEMP"):
            item |= {"type": "temporary", "surface": "temp", "claim": text,
                     "confidence": 0.5, "durability": 0.1}
        elif m.group("surface"):
            items = [x.strip() for x in text.split("options:", 1)[-1].split(";") if x.strip()]
            name = m.group("surface")
            item |= {"type": "static_surface", "surface": name,
                     "claim": f"{name} shows menu options", "closed_set_items": items,
                     "confidence": 0.9, "durability": 0.8}
        elif m.group("error"):
            prefix = "Failed" if failed else "Workaround"
            item |= {"type": "gotcha", "surface": m.group("error"),
                     "claim": f"{prefix}: {text}", "confidence": 0.8, "durability": 0.7}
        else:
            item |= {"type": "static_surface", "surface": m.group("retract"),
                     "claim": f"Retracted: {text}", "confidence": 0.8, "durability": 0.7}
        beliefs.append(item)
    return json.dumps({"beliefs": beliefs})


_CAND = re.compile(
    r"^- type=(?P<type>\S+) \| surface=(?P<surface>.*?) \| step=\d+ \| claim=(?P<claim>.*?)"
    r"(?: \| closed_set=(?P<closed>\[.*\]))?$"
)
_NEIGHBOR = re.compile(
    r"^- id=(?P<id>\S+) \| sim=\S+ \| type=\S+ \| surface=(?P<surface>.*?) \| step=\d+ \| "
    r"claim=(?P<claim>.*?)(?: \| closed_set=(?P<closed>\[.*\]))?$"
)


def _reconcile(user: str) -> str:
    lines = user.splitlines()
    cand = _CAND.match(lines[1])
    assert cand is not None, lines[1]
    neighbors = [m for ln in lines[4:] if (m := _NEIGHBOR.match(ln))]
    same = next((n for n in neighbors if n["surface"].lower() == cand["surface"].lower()), None)

    def out(op: str, target: str | None, reason: str) -> str:
        return json.dumps(
            {"op": op, "target_belief_id": target, "merged_claim": None, "reason": reason}
        )

    if same is None:
        return out("ADD", None, "new surface")
    if cand["claim"].startswith("Retracted:"):
        return out("REFUTE", same["id"], "retracted by newer evidence")
    if cand["closed"]:
        new_items = set(ast.literal_eval(cand["closed"]))
        old_items = set(ast.literal_eval(same["closed"] or "[]"))
        if new_items <= old_items:
            return out("IGNORE", None, "redundant closed set")
        return out("CONSOLIDATE", same["id"], "adds closed-set items")
    if cand["claim"] == same["claim"]:
        return out("IGNORE", None, "identical claim")
    return out("SUPERSEDE", same["id"], "newer value")


def _split_prompt(user: str) -> tuple[list[str], str]:
    memory, question = user.split("\n\nQuestion:\n", 1)
    memory = memory.removeprefix("Retrieved Memory:\n")
    return memory.split("\n\n---\n\n"), question.split("\n\n", 1)[0]


def _drift_read(user: str) -> str:
    blocks, question = _split_prompt(user)
    terms = re.findall(r"`([^`]+)`", question)
    for block in blocks:
        if not all(t in block for t in terms):
            continue
        if "durable operational policy" in question:
            if m := re.search(r"\[DURABLE POLICY:\w+\] (.+)", block):
                return f"The rule is \\boxed{{{m.group(1).strip()}}}."
        elif m := re.search(
            r"(?:is set to|changed from .+? to) (.+?)(?: \(replaces previous setting\))?\.", block
        ):
            return f"The current value is \\boxed{{{m.group(1)}}}."
    return "\\boxed{UNKNOWN}"


def _lmev2_read(user: str) -> str:
    blocks, _ = _split_prompt(user)
    if blocks == ["(no retrieved memory)"]:
        return "\\boxed{UNKNOWN}"
    first = blocks[0]
    if m := re.search(r"Closed-set (?:observed|visible) items: (.+)", first):
        return f"Observed \\boxed{{{m.group(1)}}}"
    if m := re.search(r"options: (.+)", first):
        return f"Observed \\boxed{{{m.group(1).replace(';', ',')}}}"
    return f"\\boxed{{{first.splitlines()[1][:80]}}}"


def _judge(user: str) -> str:
    ref = re.search(r"Reference answer[^\n]*:\n(.*)\n\nModel full response:\n", user, flags=re.DOTALL)
    resp = re.search(r"Model full response:\n(.*)\n\nModel extracted final answer:", user, flags=re.DOTALL)
    assert ref and resp
    words = {w for w in re.findall(r"[a-z]+", ref.group(1).lower()) if len(w) > 4}
    label = int(any(w in resp.group(1).lower() for w in words))
    return json.dumps({"label": label, "reason": "word overlap"})


_ANSWERERS = {
    "extractor": _extract,
    "reconciler": _reconcile,
    "drift_reader": _drift_read,
    "lmev2_reader": _lmev2_read,
    "judge_abstention": _judge,
    "judge_gotchas": _judge,
}


class RecordingFakeSDK:
    """Drop-in for the `openai.OpenAI` client surface used by `CachedLLMClient`."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=self)
        self.embeddings = self

    def create(self, **kwargs: Any) -> Any:
        # JSON round-trip: freeze the request exactly as sent and prove it is serializable.
        request = json.loads(json.dumps(kwargs))
        if "input" in kwargs:
            self.requests.append({"seq": len(self.requests), "kind": "embedding", "request": request})
            data = [SimpleNamespace(embedding=_embed(t)) for t in kwargs["input"]]
            return SimpleNamespace(
                data=data, usage=SimpleNamespace(total_tokens=sum(len(t) for t in kwargs["input"]) // 4)
            )

        system, user = request["messages"][0]["content"], _text_of(request["messages"][-1]["content"])
        role = next((name for name, prefix in _ROLES if system.startswith(prefix)), None)
        assert role is not None, f"Unrecognized system prompt: {system[:80]!r}"
        self.requests.append(
            {"seq": len(self.requests), "kind": "chat", "role": role, "request": request}
        )
        text = _ANSWERERS[role](user)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(
                prompt_tokens=len(json.dumps(request["messages"])) // 4,
                completion_tokens=len(text) // 4,
            ),
        )
