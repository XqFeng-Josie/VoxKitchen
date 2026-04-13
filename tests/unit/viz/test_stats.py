"""Unit tests for voxkitchen.viz.stats."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.viz.stats import compute_cutset_stats


def _prov() -> Provenance:
    return Provenance(
        source_cut_id=None,
        generated_by="test",
        stage_name="test",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
    )


def _cut(
    cid: str,
    duration: float = 1.0,
    language: str | None = None,
    speaker: str | None = None,
    snr: float | None = None,
) -> Cut:
    sups = []
    if language or speaker:
        sups.append(
            Supervision(
                id=f"{cid}-sup",
                recording_id="rec",
                start=0,
                duration=duration,
                language=language,
                speaker=speaker,
            )
        )
    metrics = {"snr": snr} if snr is not None else {}
    return Cut(
        id=cid,
        recording_id="rec",
        start=0,
        duration=duration,
        supervisions=sups,
        metrics=metrics,
        provenance=_prov(),
    )


def test_empty_cutset_stats() -> None:
    stats = compute_cutset_stats(CutSet([]))
    assert stats["count"] == 0
    assert stats["total_duration_s"] == 0
    assert stats["duration_stats"] == {}
    assert stats["languages"] == {}
    assert stats["speaker_count"] == 0


def test_basic_stats() -> None:
    cs = CutSet([_cut("a", 1.0), _cut("b", 3.0), _cut("c", 5.0)])
    stats = compute_cutset_stats(cs)
    assert stats["count"] == 3
    assert stats["total_duration_s"] == 9.0
    assert "min" in stats["duration_stats"]
    assert stats["duration_stats"]["min"] == 1.0
    assert stats["duration_stats"]["max"] == 5.0


def test_language_distribution() -> None:
    cs = CutSet(
        [
            _cut("a", language="en"),
            _cut("b", language="en"),
            _cut("c", language="zh"),
        ]
    )
    stats = compute_cutset_stats(cs)
    assert stats["languages"] == {"en": 2, "zh": 1}


def test_metrics_summary() -> None:
    cs = CutSet([_cut("a", snr=10.0), _cut("b", snr=20.0), _cut("c", snr=30.0)])
    stats = compute_cutset_stats(cs)
    assert "snr" in stats["metrics_summary"]
    snr_stats = stats["metrics_summary"]["snr"]
    assert snr_stats["min"] == 10.0
    assert snr_stats["max"] == 30.0
