"""Unit tests for ffmpeg_convert operator."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf
from voxkitchen.operators.basic.ffmpeg_convert import FfmpegConvertConfig, FfmpegConvertOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file


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
def test_ffmpeg_convert_wav_to_flac(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("convert")
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
def test_ffmpeg_convert_preserves_provenance(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("convert")
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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_convert_origin_offsets_are_absolute_in_source(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """Regression for VAD → extract chain.

    Before the fix, stage 1 (to_wav on full file) stamped origin_start=0,
    origin_end=<recording duration>, and stage 3 (extract after VAD) kept
    those stale values via ``setdefault`` — so every segment in the final
    manifest ended up with identical (0, full_duration) even though each
    VAD segment is a different slice of the source.

    The correct behaviour is: origin_start/origin_end should always
    reflect the absolute position of the cut in the ORIGINAL source file,
    computed by chaining offsets through every materializing stage.
    """
    ctx = make_run_context("convert")
    config = FfmpegConvertConfig(target_format="wav")

    # ---- Stage 1: to_wav on a full-file cut (cut.start=0). Origin bounds
    # should be (0, recording_duration).
    full_cut = _cut_from_path(mono_wav_16k)
    full_rec_duration = full_cut.duration
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    stage1_result = op.process(CutSet([full_cut]))
    op.teardown()
    stage1_cut = next(iter(stage1_result))
    assert stage1_cut.custom["origin_start"] == 0.0
    assert stage1_cut.custom["origin_end"] == round(full_rec_duration, 3)

    # ---- Simulate VAD: produce two sub-cuts at specific offsets within
    # the stage 1 output. VAD propagates custom forward unchanged, so both
    # sub-cuts still carry {origin_start: 0, origin_end: full_duration}
    # from stage 1.
    vad_cuts = [
        Cut(
            id=f"{stage1_cut.id}__svad{i}",
            recording_id=stage1_cut.recording_id,
            start=seg_start,
            duration=seg_duration,
            recording=stage1_cut.recording,
            supervisions=[],
            provenance=stage1_cut.provenance,
            custom=dict(stage1_cut.custom),
        )
        for i, (seg_start, seg_duration) in enumerate(
            [(0.2, 0.5), (0.8, 0.3)]  # two distinct VAD segments
        )
    ]

    # ---- Stage 3: extract each VAD segment into its own file.
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    stage3_result = list(op.process(CutSet(vad_cuts)))
    op.teardown()

    assert len(stage3_result) == 2
    # Segment 0: was at (0.2, 0.2+0.5=0.7) in the source
    assert stage3_result[0].custom["origin_start"] == 0.2
    assert stage3_result[0].custom["origin_end"] == 0.7
    # Segment 1: was at (0.8, 0.8+0.3=1.1) in the source — MUST differ from seg 0
    assert stage3_result[1].custom["origin_start"] == 0.8
    assert stage3_result[1].custom["origin_end"] == 1.1
