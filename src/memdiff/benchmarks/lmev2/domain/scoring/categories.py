"""LME-V2 question-type to reporting-category mapping."""

from __future__ import annotations

CATEGORY_MAP: dict[str, str] = {
    "static-environment": "static",
    "static-environment-abs": "static-abs",
    "dynamic-environment": "dynamic",
    "dynamic-environment-abs": "dynamic-abs",
    "procedure": "procedure",
    "procedure-abs": "procedure-abs",
    "errors-gotchas": "gotchas",
}
NON_ABSTENTION_CATEGORIES = ["static", "dynamic", "procedure", "gotchas"]
ABSTENTION_CATEGORIES = ["static-abs", "dynamic-abs", "procedure-abs"]
