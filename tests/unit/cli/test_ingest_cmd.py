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
