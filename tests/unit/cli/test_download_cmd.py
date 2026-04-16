"""Unit tests for vkit download CLI command."""

from __future__ import annotations

from typer.testing import CliRunner
from voxkitchen.cli.main import app

runner = CliRunner()


def test_download_help_shows_usage() -> None:
    result = runner.invoke(app, ["download", "--help"])
    assert result.exit_code == 0
    # Help text may contain ANSI escape codes from Rich
    plain = result.output.replace("\x1b[", "")
    assert "recipe" in plain.lower()
    assert "root" in plain.lower()


def test_download_unknown_recipe_exits_1() -> None:
    result = runner.invoke(app, ["download", "nonexistent_dataset", "--root", "/tmp/x"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "error" in result.output.lower()


def test_download_no_download_support_exits_1() -> None:
    # commonvoice has no download_urls, so download() should raise NotImplementedError
    result = runner.invoke(app, ["download", "commonvoice", "--root", "/tmp/x"])
    assert result.exit_code == 1
