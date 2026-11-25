"""Answer parsing and phrase matching shared by LME-V2 and DriftSim (upstream LongMemEval-V2 port)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

DEFAULT_SEPARATORS: Sequence[str] = (",", ";")


def normalize_phrase(
    text: str | None,
    *,
    lower: bool = True,
    normalize_hyphen: bool = True,
    strip_punct: bool = True,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if lower:
        text = text.lower()
    if normalize_hyphen:
        text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[,;]", " ", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def split_phrases(
    text: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    **normalize_kwargs: bool,
) -> list[str]:
    if text is None:
        return []
    separator_list = list(separators)
    if not separator_list:
        normalized = normalize_phrase(text, **normalize_kwargs)
        return [normalized] if normalized else []
    pattern = "|".join(re.escape(sep) for sep in separator_list)
    parts = re.split(pattern, text)
    normalized_parts = [normalize_phrase(part, **normalize_kwargs) for part in parts]
    return [part for part in normalized_parts if part]


def norm_phrase_set_match(
    prediction: str | None,
    answer: str | None,
    *,
    separators: Iterable[str] = DEFAULT_SEPARATORS,
    require_non_empty: bool = True,
    **normalize_kwargs: bool,
) -> bool:
    normalized_pred = normalize_phrase(prediction, **normalize_kwargs)
    answer_phrases = split_phrases(answer, separators=separators, **normalize_kwargs)
    if require_non_empty and (not normalized_pred or not answer_phrases):
        return False
    for phrase in set(answer_phrases):
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, normalized_pred) is None:
            return False
    return True


def extract_boxed_answer(text: str) -> str:
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    i = idx + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    parsed = "".join(out).strip()
    return parsed if parsed else text.strip()


def is_unknown(parsed_answer: str) -> bool:
    return parsed_answer.strip().lower() == "unknown"
