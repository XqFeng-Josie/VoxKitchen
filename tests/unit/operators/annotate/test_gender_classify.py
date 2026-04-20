"""Unit tests for gender_classify operator."""

from __future__ import annotations

try:
    import librosa  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("librosa not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.annotate.gender_classify import (
    GenderClassifyConfig,
    GenderClassifyOperator,
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


def test_gender_classify_is_registered() -> None:
    assert get_operator("gender_classify") is GenderClassifyOperator


def test_gender_classify_produces_no_audio() -> None:
    assert GenderClassifyOperator.produces_audio is False


def test_gender_classify_f0_adds_supervision(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """Running gender_classify with f0 method adds a supervision with gender."""
    cut = _cut_from_path(mono_wav_16k)
    config = GenderClassifyConfig(method="f0")
    op = GenderClassifyOperator(config, ctx=make_run_context("gender"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    # The operator appends a supervision with gender
    sups = result[0].supervisions
    assert len(sups) >= 1
    gender_sup = sups[-1]
    assert gender_sup.gender in ("m", "f", "o")
    assert gender_sup.custom is not None
    assert gender_sup.custom["gender_method"] == "f0"
