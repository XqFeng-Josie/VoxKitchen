"""Unit tests for fixed_segment operator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
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


def test_fixed_segment_is_registered() -> None:
    from voxkitchen.operators.segment.fixed_segment import FixedSegmentOperator

    assert get_operator("fixed_segment") is FixedSegmentOperator


def test_fixed_segment_splits_long_cut() -> None:
    """10s cut with 3s segments → 3 full (3s each) + 1 short (1s >= min_remaining=0.5)."""
    from voxkitchen.operators.segment.fixed_segment import FixedSegmentConfig, FixedSegmentOperator

    cut = _cut("c0", duration=10.0)
    config = FixedSegmentConfig(segment_duration=3.0, min_remaining=0.5)
    op = FixedSegmentOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut])))

    assert len(result) == 4
    assert result[0].id == "c0__seg0"
    assert result[1].id == "c0__seg1"
    assert result[2].id == "c0__seg2"
    assert result[3].id == "c0__seg3"
    assert result[0].duration == pytest.approx(3.0)
    assert result[1].duration == pytest.approx(3.0)
    assert result[2].duration == pytest.approx(3.0)
    assert result[3].duration == pytest.approx(1.0)


def test_fixed_segment_drops_short_remainder() -> None:
    """10s with 3s segments, min_remaining=2.0 → 3 full segments (drops 1s remainder)."""
    from voxkitchen.operators.segment.fixed_segment import FixedSegmentConfig, FixedSegmentOperator

    cut = _cut("c1", duration=10.0)
    config = FixedSegmentConfig(segment_duration=3.0, min_remaining=2.0)
    op = FixedSegmentOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut])))

    assert len(result) == 3
    for seg in result:
        assert seg.duration == pytest.approx(3.0)


def test_fixed_segment_preserves_recording() -> None:
    """Child cuts have the same recording object as the parent."""
    from voxkitchen.operators.segment.fixed_segment import FixedSegmentConfig, FixedSegmentOperator

    cut = _cut("c2", duration=5.0)
    config = FixedSegmentConfig(segment_duration=2.0, min_remaining=0.5)
    op = FixedSegmentOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut])))

    assert len(result) == 3  # 2s + 2s + 1s (1.0 >= 0.5)
    for child in result:
        assert child.recording_id == cut.recording_id
        assert child.recording == cut.recording
        assert child.provenance.source_cut_id == cut.id
        assert child.supervisions == []
