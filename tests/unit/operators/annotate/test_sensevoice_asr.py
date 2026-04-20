"""Unit tests for sensevoice_asr operator."""

from __future__ import annotations

try:
    from funasr import AutoModel  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("funasr not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.annotate.sensevoice_asr import (
    SenseVoiceAsrConfig,
    SenseVoiceAsrOperator,
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
# Fast (no model download)
# ---------------------------------------------------------------------------


def test_sensevoice_asr_is_registered() -> None:
    assert get_operator("sensevoice_asr") is SenseVoiceAsrOperator


def test_sensevoice_asr_class_attrs() -> None:
    assert SenseVoiceAsrOperator.device == "gpu"
    assert SenseVoiceAsrOperator.produces_audio is False
    assert SenseVoiceAsrOperator.reads_audio_bytes is True
    assert "funasr" in SenseVoiceAsrOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (downloads SenseVoice model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sensevoice_asr_transcribes(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    """Real model: a sine wave should complete without error and return 1 cut."""
    cut = _cut_from_path(mono_wav_16k)
    config = SenseVoiceAsrConfig()
    op = SenseVoiceAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1


@pytest.mark.slow
def test_sensevoice_asr_auto_language_sets_none(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """When language='auto', supervisions should have language=None (auto-detect)."""
    cut = _cut_from_path(mono_wav_16k)
    config = SenseVoiceAsrConfig(language="auto")
    op = SenseVoiceAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    for sup in result[0].supervisions:
        # When language="auto", the operator sets supervision.language=None
        assert sup.language is None
