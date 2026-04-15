"""Unit tests for resample operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)

from voxkitchen.operators.basic.resample import ResampleConfig, ResampleOperator
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
        stage_name="resample",
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


def test_resample_is_registered() -> None:
    assert get_operator("resample") is ResampleOperator


def test_resample_produces_audio() -> None:
    assert ResampleOperator.produces_audio is True


def test_resample_44k_to_16k(stereo_wav_44k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(stereo_wav_44k)])
    config = ResampleConfig(target_sr=16000)
    op = ResampleOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.samplerate == 16000


def test_resample_same_rate_passes_through(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ResampleConfig(target_sr=16000)
    op = ResampleOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    assert out_cut.recording.sampling_rate == 16000
