"""Unit tests for reverb_augment operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

try:
    from scipy.signal import fftconvolve  # noqa: F401
except ImportError:
    pytest.skip("scipy not available", allow_module_level=True)

from voxkitchen.operators.augment.reverb_augment import (
    ReverbAugmentConfig,
    ReverbAugmentOperator,
)
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


@pytest.fixture
def rir_dir(tmp_path: Path) -> Path:
    """Create a directory with a synthetic RIR (impulse + decayed tail)."""
    rdir = tmp_path / "rir"
    rdir.mkdir()
    sr = 16000
    rir = np.zeros(sr // 2, dtype=np.float32)
    rir[0] = 1.0
    rir[160] = 0.5
    rir[480] = 0.3
    decay = np.exp(-np.arange(len(rir)) / (sr * 0.05)).astype(np.float32)
    noise = np.random.RandomState(42).randn(len(rir)).astype(np.float32) * 0.01
    rir += noise * decay
    sf.write(str(rdir / "rir1.wav"), rir, sr)
    return rdir


def test_reverb_augment_is_registered() -> None:
    assert get_operator("reverb_augment") is ReverbAugmentOperator


def test_reverb_augment_produces_audio() -> None:
    assert ReverbAugmentOperator.produces_audio is True


def test_reverb_augment_output_count_matches_input(
    mono_wav_16k: Path, rir_dir: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("reverb")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ReverbAugmentConfig(rir_dir=str(rir_dir))
    op = ReverbAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    assert len(result) == 1


def test_reverb_augment_preserves_sample_rate(
    mono_wav_16k: Path, rir_dir: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("reverb")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ReverbAugmentConfig(rir_dir=str(rir_dir))
    op = ReverbAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert out_cut.recording is not None
    info = sf.info(out_cut.recording.sources[0].source)
    assert info.samplerate == 16000


def test_reverb_augment_preserves_duration(
    mono_wav_16k: Path, rir_dir: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("reverb")
    original = _cut_from_path(mono_wav_16k)
    cs = CutSet([original])
    config = ReverbAugmentConfig(rir_dir=str(rir_dir))
    op = ReverbAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert out_cut.duration == pytest.approx(original.duration, abs=0.01)


def test_reverb_augment_clips_to_valid_range(
    mono_wav_16k: Path, rir_dir: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("reverb")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ReverbAugmentConfig(rir_dir=str(rir_dir), normalize=True)
    op = ReverbAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert out_cut.recording is not None
    audio, _ = sf.read(out_cut.recording.sources[0].source, dtype="float32")
    assert np.all(audio >= -1.0)
    assert np.all(audio <= 1.0)


def test_reverb_augment_records_rir_in_custom(
    mono_wav_16k: Path, rir_dir: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("reverb")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ReverbAugmentConfig(rir_dir=str(rir_dir))
    op = ReverbAugmentOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert "rir_file" in out_cut.custom
