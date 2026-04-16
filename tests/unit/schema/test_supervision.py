"""Unit tests for voxkitchen.schema.supervision.Supervision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from voxkitchen.schema.supervision import Supervision


def test_supervision_minimal_construction() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
    )
    assert sup.id == "sup-1"
    assert sup.recording_id == "rec-1"
    assert sup.start == 0.0
    assert sup.duration == 3.5
    # optional fields default to None / empty
    assert sup.text is None
    assert sup.language is None
    assert sup.speaker is None
    assert sup.gender is None
    assert sup.age_range is None
    assert sup.custom == {}


def test_supervision_with_full_annotations() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        channel=0,
        text="hello world",
        language="en",
        speaker="spk-42",
        gender="f",
        age_range="adult",
        custom={"confidence": 0.97},
    )
    assert sup.text == "hello world"
    assert sup.language == "en"
    assert sup.speaker == "spk-42"
    assert sup.gender == "f"
    assert sup.age_range == "adult"
    assert sup.custom["confidence"] == 0.97


def test_supervision_channel_can_be_list_for_multi_channel_audio() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        channel=[0, 1],
    )
    assert sup.channel == [0, 1]


def test_supervision_rejects_invalid_gender() -> None:
    with pytest.raises(ValidationError):
        Supervision(
            id="sup-1",
            recording_id="rec-1",
            start=0.0,
            duration=3.5,
            gender="x",  # type: ignore[arg-type]
        )


def test_supervision_round_trips_through_json() -> None:
    original = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=1.25,
        duration=2.0,
        text="测试",
        language="zh",
        speaker="spk-1",
    )
    blob = original.model_dump_json()
    restored = Supervision.model_validate_json(blob)
    assert restored == original
