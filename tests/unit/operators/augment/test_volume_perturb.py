"""Unit tests for volume_perturb operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from voxkitchen.operators.augment.volume_perturb import (
    VolumePerturbConfig,
    VolumePerturbOperator,
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
        stage_name="volume",
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


def test_volume_perturb_is_registered() -> None:
    assert get_operator("volume_perturb") is VolumePerturbOperator


def test_volume_perturb_produces_audio() -> None:
    assert VolumePerturbOperator.produces_audio is True


def test_volume_perturb_output_count_matches_input(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = VolumePerturbConfig(min_gain_db=-3.0, max_gain_db=3.0)
    op = VolumePerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(result) == 1


def test_volume_perturb_preserves_sample_rate(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = VolumePerturbConfig(min_gain_db=-6.0, max_gain_db=6.0)
    op = VolumePerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    info = sf.info(out_cut.recording.sources[0].source)
    assert info.samplerate == 16000


def test_volume_perturb_preserves_duration(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = VolumePerturbConfig(min_gain_db=-6.0, max_gain_db=6.0)
    op = VolumePerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.duration == original.duration


def test_volume_perturb_clips_to_valid_range(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    # Very high gain to test clipping
    config = VolumePerturbConfig(min_gain_db=40.0, max_gain_db=40.0)
    op = VolumePerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    audio, _ = sf.read(out_cut.recording.sources[0].source, dtype="float32")
    assert np.all(audio >= -1.0)
    assert np.all(audio <= 1.0)


def test_volume_perturb_records_gain_in_custom(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = VolumePerturbConfig(min_gain_db=3.0, max_gain_db=3.0)
    op = VolumePerturbOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert "volume_gain_db" in out_cut.custom
    assert out_cut.custom["volume_gain_db"] == 3.0
