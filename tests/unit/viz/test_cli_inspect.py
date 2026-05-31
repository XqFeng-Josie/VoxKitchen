"""Tests for vkit inspect subcommands."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord, write_cuts
from voxkitchen.schema.provenance import Provenance


def _prov() -> Provenance:
    return Provenance(
        source_cut_id=None,
        generated_by="test",
        stage_name="test",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
    )


def _cut(cid: str, dur: float = 1.0) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec",
        start=0,
        duration=dur,
        supervisions=[],
        provenance=_prov(),
    )


def _header() -> HeaderRecord:
    return HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
        stage_name="test",
    )


def _write_manifest(path: Path, cuts: list[Cut]) -> None:
    write_cuts(path, _header(), iter(cuts))


def test_inspect_cuts_shows_count(tmp_path: Path) -> None:
    """inspect cuts outputs the cut count."""
    manifest = tmp_path / "cuts.jsonl.gz"
    _write_manifest(manifest, [_cut("a"), _cut("b"), _cut("c")])

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "cuts", str(manifest)])
    assert result.exit_code == 0, result.output
    assert "3" in result.output


def test_inspect_cuts_shows_duration(tmp_path: Path) -> None:
    """inspect cuts shows total duration."""
    manifest = tmp_path / "cuts.jsonl.gz"
    _write_manifest(manifest, [_cut("a", 2.0), _cut("b", 3.0)])

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "cuts", str(manifest)])
    assert result.exit_code == 0, result.output
    # Total duration is 5.0s
    assert "5.0" in result.output


def test_inspect_run_shows_stages(tmp_path: Path) -> None:
    """inspect run lists stage directory names."""
    work_dir = tmp_path / "work"
    for i, name in enumerate(["vad", "asr"]):
        sd = work_dir / f"{i:02d}_{name}"
        sd.mkdir(parents=True)
        _write_manifest(sd / "cuts.jsonl.gz", [_cut(f"{name}-cut")])
        (sd / "_SUCCESS").touch()

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", str(work_dir)])
    assert result.exit_code == 0, result.output
    assert "vad" in result.output
    assert "asr" in result.output


def test_inspect_run_shows_ok_status(tmp_path: Path) -> None:
    """inspect run marks stages with _SUCCESS as OK."""
    work_dir = tmp_path / "work"
    sd = work_dir / "00_vad"
    sd.mkdir(parents=True)
    _write_manifest(sd / "cuts.jsonl.gz", [_cut("x")])
    (sd / "_SUCCESS").touch()

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "run", str(work_dir)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_inspect_errors_no_errors(tmp_path: Path) -> None:
    """inspect errors prints 'No errors found' when no _errors.jsonl files exist."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "errors", str(work_dir)])
    assert result.exit_code == 0, result.output
    assert "No errors" in result.output


def test_inspect_errors_shows_errors(tmp_path: Path) -> None:
    """inspect errors prints error content from _errors.jsonl."""
    work_dir = tmp_path / "work"
    sd = work_dir / "00_vad"
    sd.mkdir(parents=True)
    err_line = json.dumps({"cut_id": "cut-001", "error": "timeout"})
    (sd / "_errors.jsonl").write_text(err_line + "\n")

    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "errors", str(work_dir)])
    assert result.exit_code == 0, result.output
    assert "cut-001" in result.output
    assert "timeout" in result.output


def test_inspect_subcommands_in_help() -> None:
    """inspect --help lists all three subcommands."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0, result.output
    assert "cuts" in result.output
    assert "run" in result.output
    assert "errors" in result.output


# ---------------------------------------------------------------------------
# Missing-input edge cases — each subcommand prints a one-line `error:` and
# exits with code 1 so shell scripts piping through `vkit inspect …` can
# branch on the non-zero exit.
# ---------------------------------------------------------------------------


def test_inspect_cuts_missing_manifest_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", "cuts", str(tmp_path / "absent.jsonl.gz")])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "manifest does not exist" in result.output


def test_inspect_cuts_empty_path_exits_1_without_traceback() -> None:
    """An empty path was previously routed through ``from_jsonl_gz(Path("."))`` —
    ``Path("")`` becomes ``Path(".")``, ``exists()`` returns True but it is a
    directory, so the gzip loader crashed with a traceback. Now it short-
    circuits with a friendly message.
    """
    result = CliRunner().invoke(app, ["inspect", "cuts", ""])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "no manifest path provided" in result.output
    # Crucial: no Python traceback in the output (previously had one).
    assert "Traceback" not in result.output
    assert "from_jsonl_gz" not in result.output


def test_inspect_cuts_directory_path_exits_1(tmp_path: Path) -> None:
    """Pointing at an existing directory (e.g. forgot to append the filename)
    must explain the mistake rather than crash inside the gzip loader."""
    result = CliRunner().invoke(app, ["inspect", "cuts", str(tmp_path)])
    assert result.exit_code == 1
    assert "is a directory" in result.output
    assert "Traceback" not in result.output


def test_inspect_cuts_corrupt_manifest_exits_1(tmp_path: Path) -> None:
    """An empty (or otherwise invalid) gzip file is reported as a manifest error."""
    bad = tmp_path / "corrupt.jsonl.gz"
    bad.write_bytes(b"")  # zero-byte file — not a valid manifest
    result = CliRunner().invoke(app, ["inspect", "cuts", str(bad)])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "not a valid CutSet manifest" in result.output


def test_inspect_run_missing_work_dir_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", "run", str(tmp_path / "absent")])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "work_dir does not exist" in result.output


def test_inspect_errors_missing_work_dir_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", "errors", str(tmp_path / "absent")])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_inspect_trace_missing_work_dir_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["inspect", "trace", "any-cut", "--in", str(tmp_path / "absent")]
    )
    assert result.exit_code == 1
    assert "error:" in result.output


def test_inspect_trace_unknown_cut_exits_1(tmp_path: Path) -> None:
    """An existing work_dir that does not contain the requested cut also exits 1."""
    (tmp_path / "00_stage").mkdir()  # empty stage dir is fine for this path
    result = CliRunner().invoke(app, ["inspect", "trace", "no-such-cut", "--in", str(tmp_path)])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert "not found in any stage" in result.output
