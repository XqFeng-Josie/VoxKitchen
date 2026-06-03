"""Tests for op → canonical image tag resolution."""

import pytest


def test_resolve_core_op_to_slim() -> None:
    """Ops in the 'core' operator group map to the 'slim' image tag."""
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("resample") == "slim"
    assert image_for_op("silero_vad") == "slim"
    assert image_for_op("pack_jsonl") == "slim"


def test_resolve_asr_only_op() -> None:
    """Ops in asr but not core map to 'asr'."""
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("faster_whisper_asr") == "asr"
    assert image_for_op("forced_align") == "asr"
    assert image_for_op("paraformer_asr") == "asr"


def test_resolve_diarize_only_op() -> None:
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("pyannote_diarize") == "diarize"


def test_resolve_tts_only_op() -> None:
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("tts_kokoro") == "tts"
    assert image_for_op("tts_chattts") == "tts"
    assert image_for_op("tts_cosyvoice") == "tts"


def test_resolve_fish_speech_only_op() -> None:
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("tts_fish_speech") == "fish-speech"


def test_resolve_unknown_op_raises() -> None:
    from scripts.sweep.image_resolver import UnknownOperatorError, image_for_op

    with pytest.raises(UnknownOperatorError):
        image_for_op("no_such_operator_xyz")


def test_resolve_op_not_in_any_image_group_falls_back_to_latest() -> None:
    """Ops registered locally but not in any EXPECTED_OPERATORS group fall
    back to 'latest' (the union image). Today this covers e.g. wenet_asr,
    which is intentionally excluded from the asr image's spec."""
    from scripts.sweep.image_resolver import image_for_op

    assert image_for_op("wenet_asr") == "latest"
