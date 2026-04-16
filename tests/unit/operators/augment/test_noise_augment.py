"""Unit tests for noise_augment operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voxkitchen.operators.augment.noise_augment import (
    NoiseAugmentConfig,
    NoiseAugmentOperator,
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
        stage_name="noise",
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


@pytest.fixture
def noise_dir(tmp_path: Path) -> Path:
    """Create a directory with two noise files."""
    ndir = tmp_path / "noise"
    ndir.mkdir()
    sr = 16000
    for name in ["noise1.wav", "noise2.wav"]:
        rng = np.random.RandomState(42)
        noise = (rng.randn(sr * 2) * 0.1).astype(np.float32)  # 2 seconds
        sf.write(str(ndir / name), noise, sr)
    return ndir


def test_noise_augment_is_registered() -> None:
    assert get_operator("noise_augment") is NoiseAugmentOperator


def test_noise_augment_produces_audio() -> None:
    assert NoiseAugmentOperator.produces_audio is True


def test_noise_augment_output_count_matches_input(
    mono_wav_16k: Path, noise_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = NoiseAugmentConfig(noise_dir=str(noise_dir), snr_range=[10.0, 20.0])
    op = NoiseAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(result) == 1


def test_noise_augment_preserves_sample_rate(
    mono_wav_16k: Path, noise_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = NoiseAugmentConfig(noise_dir=str(noise_dir))
    op = NoiseAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    info = sf.info(out_cut.recording.sources[0].source)
    assert info.samplerate == 16000


def test_noise_augment_preserves_duration(
    mono_wav_16k: Path, noise_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = NoiseAugmentConfig(noise_dir=str(noise_dir))
    op = NoiseAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.duration == pytest.approx(original.duration, abs=0.01)


def test_noise_augment_records_snr_in_custom(
    mono_wav_16k: Path, noise_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = NoiseAugmentConfig(noise_dir=str(noise_dir), snr_range=[15.0, 15.0])
    op = NoiseAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert "noise_snr_db" in out_cut.custom
    assert out_cut.custom["noise_snr_db"] == 15.0


def test_noise_augment_clips_to_valid_range(
    mono_wav_16k: Path, noise_dir: Path, tmp_path: Path
) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = NoiseAugmentConfig(noise_dir=str(noise_dir), snr_range=[-10.0, -10.0])
    op = NoiseAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    audio, _ = sf.read(out_cut.recording.sources[0].source, dtype="float32")
    assert np.all(audio >= -1.0)
    assert np.all(audio <= 1.0)
