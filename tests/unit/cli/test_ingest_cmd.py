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


def test_ingest_unknown_recipe_emits_friendly_error_not_traceback(
    tmp_path: Path, monkeypatch
) -> None:
    """A typo'd --recipe should print one-line error + available list, not
    a full Python KeyError traceback."""
    monkeypatch.setenv("VKIT_ALLOW_LOCAL_RUN", "1")  # silence the host-runtime warning
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "--source",
            "recipe",
            "--recipe",
            "no_such_recipe_xyz",
            "--root",
            str(tmp_path),
            "--out",
            str(tmp_path / "out.jsonl.gz"),
        ],
    )
    assert result.exit_code == 1, f"expected exit 1, got {result.exit_code}: {result.output}"
    # Should not be a Python traceback panel.
    assert "Traceback" not in result.output, result.output
    assert "KeyError" not in result.output, result.output
    # Should be the friendly form, naming the bad recipe and listing alternatives.
    assert "no_such_recipe_xyz" in result.output
    assert "librispeech" in result.output  # at least one known recipe in the suggestions
    assert "error:" in result.output.lower()
    # CliRunner with catch_exceptions=True stores leaked exceptions in
    # result.exception. The fix ensures we exit via typer.Exit, not by
    # propagating the raw KeyError.
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"expected SystemExit, got: {result.exception!r}"
    )


def test_ingest_missing_manifest_file_emits_friendly_error(tmp_path: Path) -> None:
    """A --source manifest --path that points at a non-existent file should
    render the same friendly error shape as bad-recipe, not a Python traceback."""
    ghost = tmp_path / "ghost.jsonl.gz"
    out = tmp_path / "out.jsonl.gz"
    result = CliRunner().invoke(
        app,
        [
            "ingest",
            "--source",
            "manifest",
            "--path",
            str(ghost),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1, f"got exit {result.exit_code}: {result.output}"
    assert "Traceback" not in result.output
    assert "error:" in result.output.lower()
    assert "ghost.jsonl.gz" in "".join(result.output.split())
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"expected SystemExit, got: {result.exception!r}"
    )
