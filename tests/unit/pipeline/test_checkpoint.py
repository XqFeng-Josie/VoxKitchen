"""Unit tests for voxkitchen.pipeline.checkpoint."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)


def test_stage_dir_name_is_zero_padded() -> None:
    assert stage_dir_name(0, "ingest") == "00_ingest"
    assert stage_dir_name(9, "vad") == "09_vad"
    assert stage_dir_name(10, "asr") == "10_asr"


def test_write_success_marker_creates_empty_file(tmp_path: Path) -> None:
    stage_dir = tmp_path / "00_ingest"
    stage_dir.mkdir()
    write_success_marker(stage_dir)
    marker = stage_dir / "_SUCCESS"
    assert marker.exists()
    assert marker.read_bytes() == b""


def test_is_stage_complete_requires_both_manifest_and_marker(tmp_path: Path) -> None:
    stage_dir = tmp_path / "01_vad"
    stage_dir.mkdir()

    assert not is_stage_complete(stage_dir)  # neither file
    (stage_dir / "cuts.jsonl.gz").write_bytes(b"x")
    assert not is_stage_complete(stage_dir)  # manifest only
    write_success_marker(stage_dir)
    assert is_stage_complete(stage_dir)  # both present


def test_is_stage_complete_false_if_directory_missing(tmp_path: Path) -> None:
    assert not is_stage_complete(tmp_path / "does-not-exist")


def test_find_last_completed_stage_returns_none_when_nothing_complete(tmp_path: Path) -> None:
    result = find_last_completed_stage(tmp_path, ["ingest", "vad", "asr"])
    assert result is None


def test_find_last_completed_stage_returns_highest_complete_index(tmp_path: Path) -> None:
    for i, name in enumerate(["ingest", "vad", "asr"]):
        d = tmp_path / stage_dir_name(i, name)
        d.mkdir()
    # Complete stages 0 and 1 but not 2
    (tmp_path / "00_ingest" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "00_ingest")
    (tmp_path / "01_vad" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "01_vad")

    assert find_last_completed_stage(tmp_path, ["ingest", "vad", "asr"]) == 1


def test_find_last_completed_stage_stops_at_first_incomplete(tmp_path: Path) -> None:
    """If stage 1 is incomplete but stage 2 is complete, treat stage 0 as the resume point."""
    for i, name in enumerate(["a", "b", "c"]):
        d = tmp_path / stage_dir_name(i, name)
        d.mkdir()
    (tmp_path / "00_a" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "00_a")
    # stage 1 incomplete (manifest without success)
    (tmp_path / "01_b" / "cuts.jsonl.gz").write_bytes(b"x")
    # stage 2 "complete" (but shouldn't count due to gap)
    (tmp_path / "02_c" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "02_c")

    assert find_last_completed_stage(tmp_path, ["a", "b", "c"]) == 0
