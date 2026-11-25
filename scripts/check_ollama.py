#!/usr/bin/env python3
"""Preflight check for Ollama connectivity and required chat/embedding models."""

from __future__ import annotations

import sys

from memdiff.presentation.cli.check_ollama import main

if __name__ == "__main__":
    sys.exit(main())
