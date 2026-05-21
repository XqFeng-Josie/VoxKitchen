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


def test_download_warns_on_host_install(monkeypatch) -> None:
    """`vkit download` should warn host users to use `vkit docker download`.

    The recipe-side dependencies (datasets, openslr fetchers) live in the
    Docker images, not in the lightweight PyPI launcher, so the host
    invocation typically fails. Surface the supported alternative early
    via the shared warn_if_unmanaged_runtime helper.
    """
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: False)

    # commonvoice's download() raises NotImplementedError before any network
    # traffic, so this stays fast and offline. We only care about the warning.
    result = runner.invoke(app, ["download", "commonvoice", "--root", "/tmp/x"])
    assert "vkit docker download" in result.output


def test_download_silent_inside_managed_runtime(monkeypatch) -> None:
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: True)

    result = runner.invoke(app, ["download", "commonvoice", "--root", "/tmp/x"])
    assert "vkit docker download" not in result.output
