"""Unit tests for pitch_stats operator."""

from __future__ import annotations

try:
    import pyworld  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("pyworld not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.quality.pitch_stats import (
    PitchStatsConfig,
    PitchStatsOperator,
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


# ---------------------------------------------------------------------------
# Fast (no model download)
# ---------------------------------------------------------------------------


def test_pitch_stats_is_registered() -> None:
    assert get_operator("pitch_stats") is PitchStatsOperator


def test_pitch_stats_class_attrs() -> None:
    assert PitchStatsOperator.device == "cpu"
    assert PitchStatsOperator.produces_audio is False
    assert PitchStatsOperator.reads_audio_bytes is True
    assert "pitch" in PitchStatsOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (loads pyworld)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pitch_stats_440hz_sine(mono_wav_16k: Path) -> None:
    """A 440 Hz sine wave should yield pitch_mean close to 440."""
    cut = _cut_from_path(mono_wav_16k)
    config = PitchStatsConfig()
    op = PitchStatsOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    metrics = result[0].metrics
    assert "pitch_mean" in metrics
    assert "pitch_std" in metrics
    assert isinstance(metrics["pitch_mean"], float)
    assert isinstance(metrics["pitch_std"], float)
    assert 400.0 <= metrics["pitch_mean"] <= 500.0
    assert metrics["pitch_std"] >= 0.0


@pytest.mark.slow
def test_pitch_stats_preserves_other_metrics(mono_wav_16k: Path) -> None:
    """pitch_stats merges into existing metrics without dropping them."""
    cut = _cut_from_path(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = PitchStatsConfig()
    op = PitchStatsOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "pitch_mean" in result[0].metrics


@pytest.mark.slow
def test_pitch_stats_handles_stereo(stereo_wav_44k: Path) -> None:
    """pitch_stats should handle stereo input (takes first channel)."""
    cut = _cut_from_path(stereo_wav_44k)
    config = PitchStatsConfig()
    op = PitchStatsOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert "pitch_mean" in result[0].metrics
    assert isinstance(result[0].metrics["pitch_mean"], float)
