"""Unit tests for voxkitchen.ingest.dir_scan.DirScanIngestSource."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.ingest.dir_scan import DirScanConfig, DirScanIngestSource
from voxkitchen.pipeline.context import RunContext


def _ctx(work_dir: Path) -> RunContext:
    return RunContext(
        work_dir=work_dir,
        pipeline_run_id="run-test",
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def test_dir_scan_finds_all_audio_files(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=True),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 3


def test_dir_scan_creates_cuts_with_embedded_recordings(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=True),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    for cut in cuts:
        assert cut.recording is not None
        assert cut.recording.sampling_rate > 0
        assert cut.recording.sources[0].type == "file"
        assert Path(cut.recording.sources[0].source).exists()
        assert cut.duration > 0
        assert cut.recording_id == cut.recording.id


def test_dir_scan_non_recursive(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=False),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 2


def test_dir_scan_rejects_nonexistent_directory(tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(tmp_path / "nope")),
        ctx=_ctx(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        ingest.run()


def test_dir_scan_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(empty)),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 0
