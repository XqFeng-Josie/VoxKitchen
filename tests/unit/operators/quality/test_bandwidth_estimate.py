"""Unit tests for bandwidth_estimate operator."""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("torch not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.quality.bandwidth_estimate import (
    BandwidthEstimateConfig,
    BandwidthEstimateOperator,
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


def test_bandwidth_estimate_is_registered() -> None:
    assert get_operator("bandwidth_estimate") is BandwidthEstimateOperator


def test_bandwidth_estimate_produces_no_audio() -> None:
    assert BandwidthEstimateOperator.produces_audio is False


def test_bandwidth_estimate_adds_metric(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """Running bandwidth_estimate on a 16kHz file adds bandwidth_khz metric."""
    cut = _cut_from_path(mono_wav_16k)
    config = BandwidthEstimateConfig()
    op = BandwidthEstimateOperator(config, ctx=make_run_context("bandwidth"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert "bandwidth_khz" in result[0].metrics
    assert isinstance(result[0].metrics["bandwidth_khz"], float)
    assert result[0].metrics["bandwidth_khz"] > 0.0


def test_bandwidth_estimate_preserves_other_metrics(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """bandwidth_estimate merges into existing metrics without dropping them."""
    cut = _cut_from_path(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = BandwidthEstimateConfig()
    op = BandwidthEstimateOperator(config, ctx=make_run_context("bandwidth"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "bandwidth_khz" in result[0].metrics
