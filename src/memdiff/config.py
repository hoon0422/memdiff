"""Compat shim: Settings now lives in the composition root (`memdiff.bootstrap.settings`)."""

from __future__ import annotations

from memdiff.bootstrap.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
