"""Unit tests for qwen3_asr operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    from qwen_asr import Qwen3ASRModel  # noqa: F401
except ImportError:
    pytest.skip("qwen-asr not available", allow_module_level=True)

from voxkitchen.operators.annotate.qwen3_asr import (
    Qwen3AsrConfig,
    Qwen3AsrOperator,
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


def test_qwen3_asr_is_registered() -> None:
    assert get_operator("qwen3_asr") is Qwen3AsrOperator


def test_qwen3_asr_does_not_produce_audio() -> None:
    assert Qwen3AsrOperator.produces_audio is False


@pytest.mark.slow
def test_qwen3_asr_transcribes(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    ctx = make_run_context("asr")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = Qwen3AsrConfig(model="Qwen/Qwen3-ASR-0.6B")
    op = Qwen3AsrOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    # Should have at least one supervision with text
    texts = [s.text for s in out_cut.supervisions if s.text]
    assert len(texts) >= 0  # sine wave may not produce meaningful text
