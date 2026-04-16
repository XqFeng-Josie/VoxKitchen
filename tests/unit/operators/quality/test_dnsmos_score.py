"""Unit tests for dnsmos_score operator."""

from __future__ import annotations

try:
    from speechmos import dnsmos  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("speechmos not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.quality.dnsmos_score import (
    DnsmosScoreConfig,
    DnsmosScoreOperator,
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


def test_dnsmos_score_is_registered() -> None:
    assert get_operator("dnsmos_score") is DnsmosScoreOperator


def test_dnsmos_score_class_attrs() -> None:
    assert DnsmosScoreOperator.device == "cpu"
    assert DnsmosScoreOperator.produces_audio is False
    assert DnsmosScoreOperator.reads_audio_bytes is True
    assert "dnsmos" in DnsmosScoreOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (loads ONNX models)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_dnsmos_score_adds_metrics(mono_wav_16k: Path) -> None:
    """Running dnsmos_score on a sine wave adds 4 DNSMOS metrics."""
    cut = _cut_from_path(mono_wav_16k)
    config = DnsmosScoreConfig()
    op = DnsmosScoreOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    metrics = result[0].metrics
    for key in ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "dnsmos_p808"):
        assert key in metrics, f"missing metric: {key}"
        assert isinstance(metrics[key], float)
        assert 1.0 <= metrics[key] <= 5.0


@pytest.mark.slow
def test_dnsmos_score_preserves_other_metrics(mono_wav_16k: Path) -> None:
    """dnsmos_score merges into existing metrics without dropping them."""
    cut = _cut_from_path(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = DnsmosScoreConfig()
    op = DnsmosScoreOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "dnsmos_ovrl" in result[0].metrics
