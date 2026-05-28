import pytest
from pydantic import ValidationError
from voxkitchen.datasets.catalog import DatasetEntry


def _minimal(**over):
    base = dict(
        id="librispeech",
        name="LibriSpeech",
        task=["asr"],
        languages=["en"],
        license="CC BY 4.0",
        summary="Read English audiobooks.",
        homepage="https://www.openslr.org/12",
        recommendation="Standard English ASR benchmark; clean read speech.",
    )
    base.update(over)
    return DatasetEntry(**base)


def test_minimal_entry_parses():
    e = _minimal()
    assert e.id == "librispeech"
    assert e.recipe is None and e.recommended_pipeline is None and e.hours is None


def test_optional_fields_accepted():
    e = _minimal(
        hours=960.0,
        recipe="librispeech",
        recommended_pipeline="examples/pipelines/librispeech-asr.yaml",
        domain="audiobook",
        paper="https://x",
        notes="big",
    )
    assert e.hours == 960.0 and e.recipe == "librispeech"


def test_missing_recommendation_rejected():
    with pytest.raises(ValidationError):
        DatasetEntry(
            id="x", name="X", task=["asr"], languages=["en"], license="L", summary="s", homepage="h"
        )  # no recommendation


def test_bad_task_rejected():
    with pytest.raises(ValidationError):
        _minimal(task=["transcription"])  # not in enum


def test_emotion_task_accepted():
    e = _minimal(task=["emotion"])
    assert e.task == ["emotion"]


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        _minimal(downloadable=True)  # unknown field
