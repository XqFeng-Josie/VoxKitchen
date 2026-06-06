"""Unit tests for utmos_score operator."""

from __future__ import annotations

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
# Config-only tests — run in CI without model download or network
# ---------------------------------------------------------------------------


def test_utmos_score_is_registered() -> None:
    """The operator must be importable and registered under its name."""
    assert get_operator("utmos_score") is UtmosScoreOperator


def test_utmos_score_class_attrs() -> None:
    """Contract: writes metrics.utmos; no longer requires the dnsmos extra."""
    assert UtmosScoreOperator.device == "gpu"
    assert UtmosScoreOperator.produces_audio is False
    assert UtmosScoreOperator.reads_audio_bytes is True
    # UTMOS now loads via torch.hub — speechmos/dnsmos extra is not needed.
    assert "dnsmos" not in UtmosScoreOperator.required_extras, (
        "utmos_score must not declare required_extras=['dnsmos']: the old "
        "speechmos.utmos import never worked.  Model now loads via torch.hub."
    )
    assert UtmosScoreOperator.writes == ["metrics.utmos"]


# ---------------------------------------------------------------------------
# Slow tests — load the real UTMOS22 model via torch.hub (needs network + torch)
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
