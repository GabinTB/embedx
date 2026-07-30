"""Smoke tests: package imports and CLI version work (task 00)."""

from __future__ import annotations

from typer.testing import CliRunner

import embedx
from embedx.cli import app

runner = CliRunner()


def test_version_is_str() -> None:
    assert isinstance(embedx.__version__, str)
    assert embedx.__version__


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert embedx.__version__ in result.stdout
