"""Unit tests for clipping_detect operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.quality.clipping_detect import (
    ClippingDetectConfig,
    ClippingDetectOperator,
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


def test_clipping_detect_is_registered() -> None:
    assert get_operator("clipping_detect") is ClippingDetectOperator


def test_clipping_detect_produces_no_audio() -> None:
    assert ClippingDetectOperator.produces_audio is False


def test_clipping_detect_adds_metric(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    """Running clipping_detect on a normal sine wave adds clipping_ratio metric."""
    cut = _cut_from_path(mono_wav_16k)
    config = ClippingDetectConfig()
    op = ClippingDetectOperator(config, ctx=make_run_context("clipping"))
    result = list(op.process(CutSet([cut])))

    assert len(result) == 1
    assert "clipping_ratio" in result[0].metrics
    assert isinstance(result[0].metrics["clipping_ratio"], float)
    # A 0.5-amplitude sine wave should have zero clipping at 0.99 ceiling
    assert result[0].metrics["clipping_ratio"] == 0.0


def test_clipping_detect_preserves_other_metrics(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """clipping_detect merges into existing metrics without dropping them."""
    cut = _cut_from_path(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = ClippingDetectConfig()
    op = ClippingDetectOperator(config, ctx=make_run_context("clipping"))
    result = list(op.process(CutSet([cut])))

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "clipping_ratio" in result[0].metrics
