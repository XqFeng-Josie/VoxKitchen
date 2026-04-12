"""Unit tests for quality_score_filter operator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


def _cut(cid: str, duration: float = 1.0, snr: float | None = None) -> Cut:
    metrics: dict[str, float] = {"snr": snr} if snr is not None else {}
    rec = Recording(
        id=f"rec-{cid}",
        sources=[AudioSource(type="file", channels=[0], source=f"/fake/{cid}.wav")],
        sampling_rate=16000,
        num_samples=int(16000 * duration),
        duration=duration,
        num_channels=1,
    )
    return Cut(
        id=cid,
        recording_id=rec.id,
        start=0.0,
        duration=duration,
        recording=rec,
        supervisions=[],
        metrics=metrics,
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_quality_score_filter_is_registered() -> None:
    from voxkitchen.operators.quality.quality_score_filter import QualityScoreFilterOperator

    assert get_operator("quality_score_filter") is QualityScoreFilterOperator


def test_filter_by_snr() -> None:
    """Cuts with snr 5, 15, 25 — condition 'metrics.snr > 10' keeps 15 and 25."""
    from voxkitchen.operators.quality.quality_score_filter import (
        QualityScoreFilterConfig,
        QualityScoreFilterOperator,
    )

    cuts = CutSet(
        [
            _cut("c0", snr=5.0),
            _cut("c1", snr=15.0),
            _cut("c2", snr=25.0),
        ]
    )
    config = QualityScoreFilterConfig(conditions=["metrics.snr > 10"])
    op = QualityScoreFilterOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(cuts))

    assert len(result) == 2
    assert result[0].id == "c1"
    assert result[1].id == "c2"


def test_filter_by_duration() -> None:
    """Cuts of 0.5, 2.0, 5.0 seconds — condition 'duration > 1.0' keeps 2.0 and 5.0."""
    from voxkitchen.operators.quality.quality_score_filter import (
        QualityScoreFilterConfig,
        QualityScoreFilterOperator,
    )

    cuts = CutSet(
        [
            _cut("c0", duration=0.5),
            _cut("c1", duration=2.0),
            _cut("c2", duration=5.0),
        ]
    )
    config = QualityScoreFilterConfig(conditions=["duration > 1.0"])
    op = QualityScoreFilterOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(cuts))

    assert len(result) == 2
    assert result[0].id == "c1"
    assert result[1].id == "c2"


def test_filter_multiple_conditions() -> None:
    """AND of 'metrics.snr > 10' and 'duration > 1.0' — only cuts satisfying both pass."""
    from voxkitchen.operators.quality.quality_score_filter import (
        QualityScoreFilterConfig,
        QualityScoreFilterOperator,
    )

    # c0: snr=5,  dur=2.0 → fails snr
    # c1: snr=15, dur=0.5 → fails duration
    # c2: snr=15, dur=2.0 → passes both
    # c3: snr=5,  dur=0.5 → fails both
    cuts = CutSet(
        [
            _cut("c0", duration=2.0, snr=5.0),
            _cut("c1", duration=0.5, snr=15.0),
            _cut("c2", duration=2.0, snr=15.0),
            _cut("c3", duration=0.5, snr=5.0),
        ]
    )
    config = QualityScoreFilterConfig(conditions=["metrics.snr > 10", "duration > 1.0"])
    op = QualityScoreFilterOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(cuts))

    assert len(result) == 1
    assert result[0].id == "c2"
