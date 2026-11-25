from __future__ import annotations

from pathlib import Path

from importlinter.cli import lint_imports

ROOT = Path(__file__).resolve().parents[1]


def test_import_linter_contracts_hold() -> None:
    assert lint_imports(config_filename=str(ROOT / ".importlinter")) == 0
