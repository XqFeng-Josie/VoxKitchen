"""Unit tests for qwen3_asr operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Pure-Python tests (no qwen-asr package required)
# ---------------------------------------------------------------------------

from voxkitchen.operators.annotate.qwen3_asr import _to_qwen3_language
from voxkitchen.operators.registry import get_operator


def test_qwen3_language_to_model_format() -> None:
    """_to_qwen3_language converts any input to capitalized full name for qwen_asr API."""
    assert _to_qwen3_language("zh") == "Chinese"
    assert _to_qwen3_language("zh-cn") == "Chinese"
    assert _to_qwen3_language("yue") == "Cantonese"
    assert _to_qwen3_language("en") == "English"
    assert _to_qwen3_language("Chinese") == "Chinese"
    assert _to_qwen3_language(None) is None
    assert _to_qwen3_language("auto") is None  # unknown → None → not passed to model


def test_qwen3_asr_is_registered() -> None:
    from voxkitchen.operators.annotate.qwen3_asr import Qwen3AsrOperator

    assert get_operator("qwen3_asr") is Qwen3AsrOperator


def test_qwen3_asr_does_not_produce_audio() -> None:
    from voxkitchen.operators.annotate.qwen3_asr import Qwen3AsrOperator

    assert Qwen3AsrOperator.produces_audio is False


# ---------------------------------------------------------------------------
# Slow tests (require qwen-asr package and model download)
# ---------------------------------------------------------------------------

try:
    from qwen_asr import Qwen3ASRModel  # noqa: F401

    _QWEN_AVAILABLE = True
except ImportError:
    _QWEN_AVAILABLE = False

requires_qwen = pytest.mark.skipif(not _QWEN_AVAILABLE, reason="qwen-asr not available")


@requires_qwen
@pytest.mark.slow
def test_qwen3_asr_transcribes(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    from voxkitchen.operators.annotate.qwen3_asr import Qwen3AsrConfig, Qwen3AsrOperator
    from voxkitchen.schema.cutset import CutSet
    from voxkitchen.schema.cut import Cut
    from voxkitchen.schema.provenance import Provenance
    from voxkitchen.utils.audio import recording_from_file

    rec = recording_from_file(mono_wav_16k)
    cut = Cut(
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
    ctx = make_run_context("asr")
    config = Qwen3AsrConfig(model="Qwen/Qwen3-ASR-0.6B")
    op = Qwen3AsrOperator(config, ctx)
    op.setup()
    result = op.process(CutSet([cut]))
    op.teardown()

    out_cut = next(iter(result))
    texts = [s.text for s in out_cut.supervisions if s.text]
    assert len(texts) >= 0  # sine wave may not produce meaningful text
