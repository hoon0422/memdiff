from __future__ import annotations

from enum import StrEnum
from string import Template


class PromptError(Exception):
    """A prompt is missing, malformed, or rendered with the wrong values."""


class PromptId(StrEnum):
    """Value = path under the prompts root, without the `.md` suffix."""

    EXTRACTOR_SYSTEM = "memory/belief_extractor.system"
    RECONCILER_SYSTEM = "memory/belief_reconciler.system"
    RECONCILER_USER = "memory/belief_reconciler.user"
    DRIFT_READER_SYSTEM = "benchmarks/drift_sim/reader.system"
    DRIFT_READER_USER = "benchmarks/drift_sim/reader.user"
    LMEV2_READER_SYSTEM_ENTERPRISE = "benchmarks/lmev2/reader.system.enterprise"
    LMEV2_READER_SYSTEM_WEB = "benchmarks/lmev2/reader.system.web"
    LMEV2_READER_USER = "benchmarks/lmev2/reader.user"
    JUDGE_ABSTENTION_SYSTEM = "benchmarks/lmev2/judge_abstention.system"
    JUDGE_ABSTENTION_USER = "benchmarks/lmev2/judge_abstention.user"
    JUDGE_GOTCHAS_SYSTEM = "benchmarks/lmev2/judge_gotchas.system"
    JUDGE_GOTCHAS_USER = "benchmarks/lmev2/judge_gotchas.user"

    @property
    def placeholders(self) -> frozenset[str]:
        return _PLACEHOLDERS.get(self, frozenset())


_JUDGE_USER = frozenset({"question", "reference", "response", "final_answer"})
_READER_USER = frozenset({"memory", "question"})

# Prompts not listed here are fixed text with no placeholders.
_PLACEHOLDERS: dict[PromptId, frozenset[str]] = {
    PromptId.RECONCILER_USER: frozenset({"candidate", "neighbors"}),
    PromptId.DRIFT_READER_USER: _READER_USER,
    PromptId.LMEV2_READER_USER: _READER_USER,
    PromptId.JUDGE_ABSTENTION_USER: _JUDGE_USER,
    PromptId.JUDGE_GOTCHAS_USER: _JUDGE_USER,
}


class PromptTemplate:
    """`string.Template` text whose placeholders must equal the declared set."""

    def __init__(self, prompt_id: PromptId, text: str) -> None:
        # Editors add one final newline; that single one is not part of the prompt.
        self._id = prompt_id
        self._template = Template(text.removesuffix("\n"))
        if not self._template.is_valid():
            raise PromptError(f"prompt {prompt_id.value}: invalid template syntax (stray '$'?)")
        found = frozenset(self._template.get_identifiers())
        expected = prompt_id.placeholders
        if found != expected:
            problems = []
            if expected - found:
                problems.append(f"missing placeholders {sorted(expected - found)}")
            if found - expected:
                problems.append(f"unknown placeholders {sorted(found - expected)}")
            raise PromptError(f"prompt {prompt_id.value}: " + ", ".join(problems))

    def render(self, **values: str) -> str:
        if values.keys() != self._id.placeholders:
            raise PromptError(
                f"prompt {self._id.value}: expected values for {sorted(self._id.placeholders)}, "
                f"got {sorted(values)}"
            )
        return self._template.substitute(values)
