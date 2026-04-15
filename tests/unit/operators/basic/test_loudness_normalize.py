"""Unit tests for loudness_normalize operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pyloudnorm = pytest.importorskip("pyloudnorm")

from voxkitchen.operators.basic.loudness_normalize import (  # noqa: E402
    LoudnessNormalizeConfig,
    LoudnessNormalizeOperator,
)
from voxkitchen.operators.registry import get_operator  # noqa: E402
from voxkitchen.pipeline.context import RunContext  # noqa: E402
from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.utils.audio import recording_from_file  # noqa: E402


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="loudness_normalize",
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


def test_loudness_normalize_is_registered() -> None:
    assert get_operator("loudness_normalize") is LoudnessNormalizeOperator


def test_loudness_normalize_produces_audio() -> None:
    assert LoudnessNormalizeOperator.produces_audio is True


def test_loudness_normalize_adjusts_level(mono_wav_16k: Path, tmp_path: Path) -> None:
    target_lufs = -23.0
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = LoudnessNormalizeConfig(target_lufs=target_lufs)
    op = LoudnessNormalizeOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.exists()

    audio, sr = sf.read(str(derived_path), dtype="float32")
    meter = pyloudnorm.Meter(sr)
    measured_lufs: float = meter.integrated_loudness(audio.astype(np.float64))
    assert (
        abs(measured_lufs - target_lufs) < 2.0
    ), f"Expected loudness near {target_lufs} LUFS, got {measured_lufs:.2f} LUFS"
