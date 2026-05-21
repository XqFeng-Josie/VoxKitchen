"""Tests for the vkit ingest standalone command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner
from voxkitchen.cli.main import app


def test_ingest_dir_writes_manifest(audio_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl.gz"
    runner = CliRunner()
    result = runner.invoke(
        app, ["ingest", "--source", "dir", "--root", str(audio_dir), "--out", str(out)]
    )
    assert result.exit_code == 0
    assert out.exists()
    from voxkitchen.schema.io import read_cuts

    cuts = list(read_cuts(out))
    assert len(cuts) == 3  # audio_dir has 3 files


def test_ingest_missing_root_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--source", "dir", "--out", str(tmp_path / "x.gz")])
    assert result.exit_code == 1


def test_ingest_unknown_source_exits_1(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "--source", "alien", "--out", str(tmp_path / "x.gz")])
    assert result.exit_code == 1


def test_ingest_dir_is_silent_about_managed_runtime(
    audio_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    """`ingest --source dir` works with only the lightweight launcher deps,
    so the host-warning must NOT fire for it.
    """
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: False)

    result = CliRunner().invoke(
        app,
        ["ingest", "--source", "dir", "--root", str(audio_dir), "--out", str(tmp_path / "out.gz")],
    )
    assert result.exit_code == 0
    assert "vkit docker download" not in result.output


def test_ingest_recipe_warns_on_host_install(tmp_path: Path, monkeypatch) -> None:
    """`ingest --source recipe` needs recipe-specific deps that ship in
    Docker images. Surface the warning when invoked from a host install.
    """
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: False)

    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "--source",
            "recipe",
            "--recipe",
            "librispeech",
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.gz"),
        ],
    )
    # The recipe-prepare step will likely fail because /tmp is empty; we
    # only assert on the warning being printed first. Rich wraps the line
    # at narrow CliRunner widths, so collapse whitespace before matching.
    collapsed = " ".join(result.output.split())
    assert "warning:" in collapsed
    assert "vkit ingest --source recipe" in collapsed
    assert "vkit docker download" in collapsed


def test_ingest_error_messages_use_error_prefix(tmp_path: Path) -> None:
    """Inline error strings carry the same `error:` prefix the rest of the CLI uses."""
    out = str(tmp_path / "x.gz")
    runner = CliRunner()

    missing_root = runner.invoke(app, ["ingest", "--source", "dir", "--out", out])
    assert "error:" in missing_root.output
    assert "--root required" in missing_root.output

    missing_path = runner.invoke(app, ["ingest", "--source", "manifest", "--out", out])
    assert "error:" in missing_path.output

    bad_source = runner.invoke(app, ["ingest", "--source", "alien", "--out", out])
    assert "error:" in bad_source.output
    assert "unknown source" in bad_source.output
