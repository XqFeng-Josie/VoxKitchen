"""Smoke tests: `vkit --help` runs and exits cleanly."""

from __future__ import annotations

from typer.testing import CliRunner

from voxkitchen.cli.main import app


def test_top_level_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vkit" in result.output.lower()


def test_six_top_level_commands_are_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    expected_commands = {"init", "ingest", "validate", "run", "inspect", "viz"}
    for cmd in expected_commands:
        assert cmd in result.output, f"command '{cmd}' missing from --help output"


def test_init_and_ingest_are_real_commands() -> None:
    runner = CliRunner()
    # `init --help` should exit 0 (real command, not a placeholder)
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    # `ingest --help` should exit 0 (real command, not a placeholder)
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
