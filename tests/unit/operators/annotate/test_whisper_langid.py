"""Unit tests for whisper_langid operator."""

from __future__ import annotations

_has_backend = False
try:
    import whisper  # noqa: F401

    _has_backend = True
except ImportError:
    pass

if not _has_backend:
    try:
        import faster_whisper  # noqa: F401

        _has_backend = True
    except ImportError:
        pass

import pytest  # noqa: E402

if not _has_backend:
    pytest.skip("neither openai-whisper nor faster-whisper available", allow_module_level=True)

from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

from voxkitchen.operators.annotate.whisper_langid import (  # noqa: E402
    WhisperLangidConfig,
    WhisperLangidOperator,
)
from voxkitchen.operators.registry import get_operator  # noqa: E402
from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.utils.audio import recording_from_file  # noqa: E402


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


def test_whisper_langid_is_registered() -> None:
    assert get_operator("whisper_langid") is WhisperLangidOperator


def test_whisper_langid_class_attrs() -> None:
    assert WhisperLangidOperator.device == "gpu"
    assert WhisperLangidOperator.produces_audio is False
    assert WhisperLangidOperator.reads_audio_bytes is True
    assert "whisper" in WhisperLangidOperator.required_extras
    # Config defaults
    cfg = WhisperLangidConfig()
    assert cfg.backend == "auto"
    assert cfg.model == "tiny"


# ---------------------------------------------------------------------------
# Slow (downloads whisper-tiny ~75 MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_whisper_langid_detects_language(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """Real tiny model: should return 1 cut with a supervision containing a non-empty language."""
    cut = _cut_from_path(mono_wav_16k)
    config = WhisperLangidConfig(model="tiny")
    op = WhisperLangidOperator(config, ctx=make_run_context("langid"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    langid_sups = [s for s in result[0].supervisions if s.id.endswith("__langid")]
    assert len(langid_sups) == 1
    assert isinstance(langid_sups[0].language, str)
    assert len(langid_sups[0].language) > 0
