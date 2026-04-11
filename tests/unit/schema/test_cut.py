"""Unit tests for voxkitchen.schema.cut.Cut."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _make_provenance(source_id: str | None = "cut-parent") -> Provenance:
    return Provenance(
        source_cut_id=source_id,
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )


def test_cut_minimal_construction() -> None:
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut.id == "cut-1"
    assert cut.recording_id == "rec-1"
    assert cut.start == 0.0
    assert cut.duration == 3.5
    assert cut.supervisions == []
    assert cut.metrics == {}
    assert cut.custom == {}


def test_cut_with_supervisions_and_metrics() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        text="hello world",
    )
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[sup],
        metrics={"snr": 18.3, "clip_rate": 0.001},
        provenance=_make_provenance(),
    )
    assert len(cut.supervisions) == 1
    assert cut.supervisions[0].text == "hello world"
    assert cut.metrics["snr"] == 18.3


def test_cut_channel_may_be_int_list_or_none() -> None:
    cut_single = Cut(
        id="c1",
        recording_id="r1",
        start=0.0,
        duration=1.0,
        channel=0,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut_single.channel == 0

    cut_multi = Cut(
        id="c2",
        recording_id="r1",
        start=0.0,
        duration=1.0,
        channel=[0, 1],
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut_multi.channel == [0, 1]


def test_cut_rejects_missing_provenance() -> None:
    with pytest.raises(ValidationError):
        Cut(  # type: ignore[call-arg]
            id="cut-1",
            recording_id="rec-1",
            start=0.0,
            duration=3.5,
            supervisions=[],
        )


def test_cut_round_trips_through_json() -> None:
    original = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[
            Supervision(
                id="sup-1",
                recording_id="rec-1",
                start=0.0,
                duration=3.5,
                text="hello world",
                language="en",
            )
        ],
        metrics={"snr": 18.3},
        provenance=_make_provenance(),
        custom={"split": "train"},
    )
    blob = original.model_dump_json()
    restored = Cut.model_validate_json(blob)
    assert restored == original
