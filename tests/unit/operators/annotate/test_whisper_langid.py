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

if not _has_backend:
    import pytest

    pytest.skip("neither openai-whisper nor faster-whisper available", allow_module_level=True)

from voxkitchen.operators.annotate.whisper_langid import (
    WhisperLangidOperator,
)
from voxkitchen.operators.registry import get_operator


def test_whisper_langid_is_registered() -> None:
    assert get_operator("whisper_langid") is WhisperLangidOperator
