"""Unit tests for sensevoice_asr operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pure-Python tests (no funasr package required)
# ---------------------------------------------------------------------------

from voxkitchen.operators.annotate.sensevoice_asr import _strip_sensevoice_tags
from voxkitchen.operators.registry import get_operator


def test_strip_sensevoice_tags() -> None:
    assert _strip_sensevoice_tags("<|zh|><|HAPPY|><|Speech|><|withitn|>参观海洋馆。") == "参观海洋馆。"
    assert _strip_sensevoice_tags("<|zh|><|NEUTRAL|><|BGM|><|withitn|>hello") == "hello"
    assert _strip_sensevoice_tags("no tags here") == "no tags here"
    assert _strip_sensevoice_tags("") == ""


def test_parse_sensevoice_output() -> None:
    from voxkitchen.operators.annotate.sensevoice_asr import _parse_sensevoice_output

    text, lang, emotion, event = _parse_sensevoice_output(
        "<|zh|><|HAPPY|><|Speech|><|withitn|>参观海洋馆。"
    )
    assert text == "参观海洋馆。"
    assert lang == "chinese"
    assert emotion == "happy"
    assert event == "Speech"

    text, lang, emotion, event = _parse_sensevoice_output(
        "<|en|><|NEUTRAL|><|BGM|><|withitn|>Hello world."
    )
    assert lang == "english"
    assert emotion == "neutral"
    assert event == "BGM"

    # No tags — all metadata None
    text, lang, emotion, event = _parse_sensevoice_output("plain text")
    assert text == "plain text"
    assert lang is None and emotion is None and event is None


def test_sensevoice_asr_is_registered() -> None:
    from voxkitchen.operators.annotate.sensevoice_asr import SenseVoiceAsrOperator

    assert get_operator("sensevoice_asr") is SenseVoiceAsrOperator


def test_sensevoice_asr_class_attrs() -> None:
    from voxkitchen.operators.annotate.sensevoice_asr import SenseVoiceAsrOperator

    assert SenseVoiceAsrOperator.device == "gpu"
    assert SenseVoiceAsrOperator.produces_audio is False
    assert SenseVoiceAsrOperator.reads_audio_bytes is True
    assert "funasr" in SenseVoiceAsrOperator.required_extras


# ---------------------------------------------------------------------------
# Slow tests (require funasr package and model download)
# ---------------------------------------------------------------------------

try:
    from funasr import AutoModel  # noqa: F401

    _FUNASR_AVAILABLE = True
except ImportError:
    _FUNASR_AVAILABLE = False

requires_funasr = pytest.mark.skipif(not _FUNASR_AVAILABLE, reason="funasr not available")


def _cut_from_path(audio_path: Path) -> "Cut":  # type: ignore[name-defined]  # noqa: F821
    from voxkitchen.schema.cut import Cut
    from voxkitchen.schema.provenance import Provenance
    from voxkitchen.utils.audio import recording_from_file

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


@requires_funasr
@pytest.mark.slow
def test_sensevoice_asr_transcribes(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    from voxkitchen.operators.annotate.sensevoice_asr import SenseVoiceAsrConfig, SenseVoiceAsrOperator
    from voxkitchen.schema.cutset import CutSet

    cut = _cut_from_path(mono_wav_16k)
    config = SenseVoiceAsrConfig()
    op = SenseVoiceAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()
    assert len(result) == 1


@requires_funasr
@pytest.mark.slow
def test_sensevoice_asr_output_has_no_tags(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    """Supervision text must not contain raw SenseVoice tags after processing."""
    from voxkitchen.operators.annotate.sensevoice_asr import SenseVoiceAsrConfig, SenseVoiceAsrOperator
    from voxkitchen.schema.cutset import CutSet
    import re

    cut = _cut_from_path(mono_wav_16k)
    config = SenseVoiceAsrConfig()
    op = SenseVoiceAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    tag_re = re.compile(r"<\|[^|]*\|>")
    for cut_out in result:
        for sup in cut_out.supervisions:
            assert sup.text is None or not tag_re.search(sup.text), (
                f"Supervision text contains raw SenseVoice tags: {sup.text!r}"
            )


@requires_funasr
@pytest.mark.slow
def test_sensevoice_asr_auto_language_sets_none(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    from voxkitchen.operators.annotate.sensevoice_asr import SenseVoiceAsrConfig, SenseVoiceAsrOperator
    from voxkitchen.schema.cutset import CutSet

    cut = _cut_from_path(mono_wav_16k)
    config = SenseVoiceAsrConfig(language="auto")
    op = SenseVoiceAsrOperator(config, ctx=make_run_context("asr"))
    op.setup()
    result = list(op.process(CutSet([cut])))
    op.teardown()

    assert len(result) == 1
    for sup in result[0].supervisions:
        assert sup.language is None
