"""Unit tests for voxkitchen.schema.recording.Recording and AudioSource."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxkitchen.schema.recording import AudioSource, Recording


def test_audio_source_file_construction() -> None:
    src = AudioSource(type="file", channels=[0], source="/data/foo.wav")
    assert src.type == "file"
    assert src.channels == [0]
    assert src.source == "/data/foo.wav"


def test_audio_source_url_construction() -> None:
    src = AudioSource(
        type="url",
        channels=[0, 1],
        source="https://example.com/audio.flac",
    )
    assert src.type == "url"


def test_audio_source_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AudioSource(type="carrier_pigeon", channels=[0], source="wat")  # type: ignore[arg-type]


def test_recording_minimal_construction() -> None:
    rec = Recording(
        id="librispeech-1089-134686-0001",
        sources=[AudioSource(type="file", channels=[0], source="/data/foo.wav")],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
        num_channels=1,
    )
    assert rec.id == "librispeech-1089-134686-0001"
    assert rec.duration == 10.0
    assert rec.checksum is None
    assert rec.custom == {}


def test_recording_accepts_checksum_and_custom() -> None:
    rec = Recording(
        id="rec-1",
        sources=[AudioSource(type="file", channels=[0], source="/data/foo.wav")],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
        num_channels=1,
        checksum="a" * 64,
        custom={"origin": "librispeech", "subset": "train-clean-100"},
    )
    assert rec.checksum == "a" * 64
    assert rec.custom["origin"] == "librispeech"


def test_recording_round_trips_through_json() -> None:
    original = Recording(
        id="rec-1",
        sources=[
            AudioSource(type="file", channels=[0], source="/data/left.wav"),
            AudioSource(type="file", channels=[1], source="/data/right.wav"),
        ],
        sampling_rate=48000,
        num_samples=480000,
        duration=10.0,
        num_channels=2,
    )
    blob = original.model_dump_json()
    restored = Recording.model_validate_json(blob)
    assert restored == original


def test_recording_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Recording(  # type: ignore[call-arg]
            id="rec-1",
            sources=[],
            sampling_rate=16000,
            num_samples=160000,
            num_channels=1,
        )
