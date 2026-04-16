"""Unit tests for speed_perturb operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    pytest.skip("torchaudio not available", allow_module_level=True)

from voxkitchen.operators.augment.speed_perturb import (
    SpeedPerturbConfig,
    SpeedPerturbOperator,
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
        stage_name="speed",
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


def test_speed_perturb_is_registered() -> None:
    assert get_operator("speed_perturb") is SpeedPerturbOperator


def test_speed_perturb_produces_audio() -> None:
    assert SpeedPerturbOperator.produces_audio is True


def test_speed_perturb_generates_one_cut_per_factor(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = SpeedPerturbConfig(factors=[0.9, 1.1])
    op = SpeedPerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(result) == 2  # one cut per factor


def test_speed_perturb_changes_duration(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = SpeedPerturbConfig(factors=[0.9])
    op = SpeedPerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    # Factor 0.9 -> audio becomes longer (1/0.9 ~ 1.11x)
    assert out_cut.duration > original.duration * 1.05


def test_speed_perturb_preserves_sample_rate(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = SpeedPerturbConfig(factors=[1.1])
    op = SpeedPerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    info = sf.info(out_cut.recording.sources[0].source)
    assert info.samplerate == 16000


def test_speed_perturb_factor_1_preserves_duration(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = SpeedPerturbConfig(factors=[1.0])
    op = SpeedPerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.duration == pytest.approx(original.duration, abs=0.01)
