"""Unit tests for whisper_openai_asr operator."""

from __future__ import annotations

try:
    import whisper  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("openai-whisper not available", allow_module_level=True)

from voxkitchen.operators.annotate.whisper_openai_asr import (
    WhisperOpenaiAsrOperator,
)
from voxkitchen.operators.registry import get_operator


def test_whisper_openai_asr_is_registered() -> None:
    assert get_operator("whisper_openai_asr") is WhisperOpenaiAsrOperator
