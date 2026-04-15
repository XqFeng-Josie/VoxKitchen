"""Unit tests for speech_enhance operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import df  # noqa: F401
except ImportError:
    pytest.skip("deepfilternet not available", allow_module_level=True)

from voxkitchen.operators.annotate.speech_enhance import (
    SpeechEnhanceConfig,
    SpeechEnhanceOperator,
)
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
        stage_name="enhance",
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


def test_speech_enhance_is_registered() -> None:
    assert get_operator("speech_enhance") is SpeechEnhanceOperator


def test_speech_enhance_produces_audio() -> None:
    assert SpeechEnhanceOperator.produces_audio is True


@pytest.mark.slow
def test_speech_enhance_preserves_sample_rate(
    mono_wav_16k: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = SpeechEnhanceConfig()
    op = SpeechEnhanceOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert out_cut.recording is not None
    info = sf.info(out_cut.recording.sources[0].source)
    assert info.samplerate == 16000


@pytest.mark.slow
def test_speech_enhance_preserves_duration(
    mono_wav_16k: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = SpeechEnhanceConfig()
    op = SpeechEnhanceOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert out_cut.duration == pytest.approx(original.duration, abs=0.05)
