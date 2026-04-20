"""Unit tests for whisper_openai_asr operator."""

from __future__ import annotations

try:
    import whisper  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("openai-whisper not available", allow_module_level=True)

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.annotate.whisper_openai_asr import (
    WhisperOpenaiAsrConfig,
    WhisperOpenaiAsrOperator,
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


def test_whisper_openai_asr_is_registered() -> None:
    assert get_operator("whisper_openai_asr") is WhisperOpenaiAsrOperator


def test_whisper_openai_asr_class_attrs() -> None:
    assert WhisperOpenaiAsrOperator.device == "gpu"
    assert WhisperOpenaiAsrOperator.produces_audio is False
    assert WhisperOpenaiAsrOperator.reads_audio_bytes is True
    assert "whisper" in WhisperOpenaiAsrOperator.required_extras
    # Config defaults
    cfg = WhisperOpenaiAsrConfig()
    assert cfg.model == "tiny"
    assert cfg.language is None
    assert cfg.beam_size == 5
    assert cfg.fp16 is True


# ---------------------------------------------------------------------------
# Slow (downloads whisper-tiny ~75 MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_whisper_openai_asr_transcribes(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """Real tiny model on CPU: a sine wave should complete without error."""
    cut = _cut_from_path(mono_wav_16k)
    config = WhisperOpenaiAsrConfig(model="tiny", fp16=False)
    op = WhisperOpenaiAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    assert isinstance(result[0].supervisions, list)
