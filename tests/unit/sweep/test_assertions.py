"""Tests for the operator sweep's assertion module."""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _write_pack_stage(
    work_dir: Path, *, stage_name: str = "01_pack", cuts: list[dict] | None = None
) -> Path:
    """Write a minimal pack-stage manifest into work_dir for assertion testing."""
    stage_dir = work_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest = stage_dir / "cuts.jsonl.gz"
    header = {
        "__type__": "voxkitchen.header",
        "schema_version": "0.1",
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "pipeline_run_id": "test",
        "stage_name": stage_name,
    }
    with gzip.open(manifest, "wt", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for cut in cuts or []:
            f.write(json.dumps({"__type__": "cut", **cut}) + "\n")
    return stage_dir


def _make_cut(**overrides) -> dict:
    """Build a valid Cut dict for tests."""
    base = {
        "id": "c1",
        "recording_id": "c1",
        "start": 0.0,
        "duration": 1.0,
        "channel": None,
        "recording": {
            "id": "c1",
            "sources": [{"type": "file", "channels": [0], "source": "x.wav"}],
            "sampling_rate": 16000,
            "num_samples": 16000,
            "duration": 1.0,
            "num_channels": 1,
            "checksum": None,
            "custom": {},
        },
        "supervisions": [],
        "metrics": {},
        "provenance": {
            "source_cut_id": None,
            "generated_by": "test",
            "stage_name": "ingest",
            "created_at": "2026-01-01T00:00:00Z",
            "pipeline_run_id": "t",
        },
        "custom": {},
    }
    base.update(overrides)
    return base


def test_default_smoke_passes_with_cuts(tmp_path: Path) -> None:
    from scripts.sweep.assertions import default_smoke_assertion

    _write_pack_stage(tmp_path, cuts=[_make_cut(), _make_cut(id="c2", recording_id="c2")])
    ok, msg = default_smoke_assertion(tmp_path, "")
    assert ok
    assert "2 cuts" in msg


def test_default_smoke_fails_with_no_cuts(tmp_path: Path) -> None:
    from scripts.sweep.assertions import default_smoke_assertion

    _write_pack_stage(tmp_path, cuts=[])
    ok, msg = default_smoke_assertion(tmp_path, "")
    assert not ok
    assert "0 cuts" in msg


def test_resample_assertion_checks_sample_rate(tmp_path: Path) -> None:
    from scripts.sweep.assertions import assert_resample_target_sr

    good = _make_cut(
        recording={
            **_make_cut()["recording"],
            "sampling_rate": 16000,
        }
    )
    _write_pack_stage(tmp_path, cuts=[good])
    ok, _ = assert_resample_target_sr(tmp_path, "")
    assert ok

    bad = _make_cut(recording={**_make_cut()["recording"], "sampling_rate": 48000})
    work2 = tmp_path / "work2"
    _write_pack_stage(work2, cuts=[bad])
    ok, msg = assert_resample_target_sr(work2, "")
    assert not ok
    assert "16000" in msg


def test_vad_segments_passes_with_short_segments(tmp_path: Path) -> None:
    from scripts.sweep.assertions import assert_vad_segments

    cuts = [_make_cut(id=f"c{i}", recording_id=f"c{i}", duration=5.0) for i in range(3)]
    _write_pack_stage(tmp_path, cuts=cuts)
    ok, msg = assert_vad_segments(tmp_path, "")
    assert ok
    assert "3 VAD segments" in msg


def test_asr_nonempty_requires_supervision_text(tmp_path: Path) -> None:
    from scripts.sweep.assertions import assert_asr_nonempty

    cut_with_text = _make_cut(
        supervisions=[
            {
                "id": "s0",
                "recording_id": "c1",
                "start": 0.0,
                "duration": 1.0,
                "channel": None,
                "text": "hello world",
                "language": None,
                "speaker": None,
                "gender": None,
                "age_range": None,
                "custom": {},
            }
        ]
    )
    _write_pack_stage(tmp_path, cuts=[cut_with_text])
    ok, msg = assert_asr_nonempty(tmp_path, "")
    assert ok
    assert "2 words" in msg

    work2 = tmp_path / "work2"
    _write_pack_stage(work2, cuts=[_make_cut()])  # no supervisions
    ok, _ = assert_asr_nonempty(work2, "")
    assert not ok


def test_metric_written_factory(tmp_path: Path) -> None:
    from scripts.sweep.assertions import assert_metric_written

    cut = _make_cut(metrics={"snr": 12.5})
    _write_pack_stage(tmp_path, cuts=[cut])
    ok, _ = assert_metric_written("snr")(tmp_path, "")
    assert ok

    work2 = tmp_path / "work2"
    _write_pack_stage(work2, cuts=[_make_cut()])
    ok, _ = assert_metric_written("snr")(work2, "")
    assert not ok


def test_normalize_text_strips_markup(tmp_path: Path) -> None:
    from scripts.sweep.assertions import assert_normalize_text_strips

    good = _make_cut(
        supervisions=[
            {
                "id": "s0",
                "recording_id": "c1",
                "start": 0.0,
                "duration": 1.0,
                "channel": None,
                "text": "hello world",
                "language": None,
                "speaker": None,
                "gender": None,
                "age_range": None,
                "custom": {},
            }
        ]
    )
    _write_pack_stage(tmp_path, cuts=[good])
    ok, _ = assert_normalize_text_strips(tmp_path, "")
    assert ok

    bad = _make_cut(
        supervisions=[
            {
                "id": "s0",
                "recording_id": "c1",
                "start": 0.0,
                "duration": 1.0,
                "channel": None,
                "text": "<|en|> hello world",
                "language": None,
                "speaker": None,
                "gender": None,
                "age_range": None,
                "custom": {},
            }
        ]
    )
    work2 = tmp_path / "work2"
    _write_pack_stage(work2, cuts=[bad])
    ok, msg = assert_normalize_text_strips(work2, "")
    assert not ok
    assert "markup" in msg.lower() or "double" in msg.lower()


def test_loudness_normalize_uses_smoke_assertion() -> None:
    """The sweep pipeline plan called for assert_metric_written('loudness_lufs')
    but the operator doesn't write that metric — verified in commit 23afe82.
    Pin the smoke fallback so an accidental "upgrade" doesn't slip through."""
    from scripts.sweep.assertions import ASSERTIONS, default_smoke_assertion

    assert ASSERTIONS["loudness_normalize"] is default_smoke_assertion, (
        "loudness_normalize must use default_smoke_assertion until the "
        "operator actually writes a loudness metric. If you're upgrading "
        "this, also confirm voxkitchen/operators/basic/loudness_normalize.py "
        "writes a loudness_lufs entry to cut.metrics."
    )


def test_assertion_returns_false_on_missing_manifest(tmp_path: Path) -> None:
    """If the pipeline somehow produced no pack stage, assertion fails cleanly."""
    from scripts.sweep.assertions import default_smoke_assertion

    ok, msg = default_smoke_assertion(tmp_path, "")
    assert not ok
    assert "0 cuts" in msg


def test_read_final_cuts_logs_warning_on_corrupt_manifest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt manifest must produce a WARNING log entry and []."""
    from scripts.sweep.assertions import _read_final_cuts

    # Build a stage dir with a gzip file that's NOT a valid header+cut stream.
    stage_dir = tmp_path / "01_pack"
    stage_dir.mkdir()
    bad = stage_dir / "cuts.jsonl.gz"
    with gzip.open(bad, "wt") as f:
        f.write("not a valid voxkitchen manifest line\n")

    with caplog.at_level("WARNING", logger="scripts.sweep.assertions"):
        result = _read_final_cuts(tmp_path)

    assert result == []
    # The warning must name the manifest path and the exception type.
    matching = [r for r in caplog.records if r.name == "scripts.sweep.assertions"]
    assert matching, f"expected a WARNING from scripts.sweep.assertions, got: {caplog.records}"
    msg = str(matching[0].getMessage())
    assert str(bad) in msg or "cuts.jsonl.gz" in msg
