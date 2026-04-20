"""Unit tests for codec_tokenize operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import encodec  # noqa: F401
except ImportError:
    pytest.skip("encodec not available", allow_module_level=True)

from voxkitchen.operators.annotate.codec_tokenize import (
    CodecTokenizeConfig,
    CodecTokenizeOperator,
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
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_codec_tokenize_is_registered() -> None:
    assert get_operator("codec_tokenize") is CodecTokenizeOperator


def test_codec_tokenize_does_not_produce_audio() -> None:
    assert CodecTokenizeOperator.produces_audio is False


@pytest.mark.slow
def test_codec_tokenize_encodec_produces_tokens(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("codec")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = CodecTokenizeConfig(backend="encodec")
    op = CodecTokenizeOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    tokens = out.custom.get("codec_tokens")
    assert tokens is not None
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(layer, list) for layer in tokens)
    assert all(isinstance(t, int) for t in tokens[0])
    assert out.custom["codec_backend"] == "encodec"
