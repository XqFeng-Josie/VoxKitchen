"""Unit tests for utmos_score operator."""

from __future__ import annotations

try:
    from speechmos import utmos  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("speechmos not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.operators.quality.utmos_score import (
    UtmosScoreConfig,
    UtmosScoreOperator,
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


def test_utmos_score_is_registered() -> None:
    assert get_operator("utmos_score") is UtmosScoreOperator


def test_utmos_score_class_attrs() -> None:
    assert UtmosScoreOperator.device == "cpu"
    assert UtmosScoreOperator.produces_audio is False
    assert UtmosScoreOperator.reads_audio_bytes is True
    assert "dnsmos" in UtmosScoreOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (loads UTMOS model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_utmos_score_adds_metric(mono_wav_16k: Path) -> None:
    """Running utmos_score on a sine wave adds a utmos metric."""
    cut = _cut_from_path(mono_wav_16k)
    config = UtmosScoreConfig()
    op = UtmosScoreOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert "utmos" in result[0].metrics
    assert isinstance(result[0].metrics["utmos"], float)
    assert 1.0 <= result[0].metrics["utmos"] <= 5.0


@pytest.mark.slow
def test_utmos_score_preserves_other_metrics(mono_wav_16k: Path) -> None:
    """utmos_score merges into existing metrics without dropping them."""
    cut = _cut_from_path(mono_wav_16k)
    cut = cut.model_copy(update={"metrics": {"existing": 1.0}})
    config = UtmosScoreConfig()
    op = UtmosScoreOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert result[0].metrics["existing"] == 1.0
    assert "utmos" in result[0].metrics
