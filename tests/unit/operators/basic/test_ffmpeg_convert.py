"""Unit tests for ffmpeg_convert operator."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf
from voxkitchen.operators.basic.ffmpeg_convert import FfmpegConvertConfig, FfmpegConvertOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="convert",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _cut_from_path(audio_path: Path) -> Cut:
    rec = recording_from_file(audio_path)
    return Cut(
        id=f"cut-{rec.id}",
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
        recording=rec,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_ffmpeg_convert_is_registered() -> None:
    assert get_operator("ffmpeg_convert") is FfmpegConvertOperator


def test_ffmpeg_convert_produces_audio() -> None:
    assert FfmpegConvertOperator.produces_audio is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_convert_wav_to_flac(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = FfmpegConvertConfig(target_format="flac")
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(result) == 1
    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.suffix == ".flac"
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.samplerate == 16000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_convert_preserves_provenance(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    original_cut = _cut_from_path(mono_wav_16k)
    cs = CutSet([original_cut])
    config = FfmpegConvertConfig(target_format="wav")
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.provenance.source_cut_id == original_cut.id
    assert out_cut.provenance.generated_by.startswith("ffmpeg_convert")
