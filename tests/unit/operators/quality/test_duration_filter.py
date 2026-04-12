"""Unit tests for duration_filter operator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


def _cut(cid: str, duration: float = 10.0) -> Cut:
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
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_duration_filter_is_registered() -> None:
    from voxkitchen.operators.quality.duration_filter import DurationFilterOperator

    assert get_operator("duration_filter") is DurationFilterOperator


def test_duration_filter_keeps_cuts_in_range() -> None:
    """Cuts of 0.5s, 2s, 5s, 10s with min=1 max=6 → keeps only 2s and 5s."""
    from voxkitchen.operators.quality.duration_filter import (
        DurationFilterConfig,
        DurationFilterOperator,
    )

    cuts = CutSet(
        [
            _cut("c0", 0.5),
            _cut("c1", 2.0),
            _cut("c2", 5.0),
            _cut("c3", 10.0),
        ]
    )
    config = DurationFilterConfig(min_duration=1.0, max_duration=6.0)
    op = DurationFilterOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(cuts))

    assert len(result) == 2
    assert result[0].id == "c1"
    assert result[1].id == "c2"


def test_duration_filter_defaults_keep_all() -> None:
    """Default config (no min/max) → all cuts pass."""
    from voxkitchen.operators.quality.duration_filter import (
        DurationFilterConfig,
        DurationFilterOperator,
    )

    cuts = CutSet([_cut("c0", 0.1), _cut("c1", 100.0), _cut("c2", 0.0)])
    config = DurationFilterConfig()
    op = DurationFilterOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(cuts))

    assert len(result) == 3
