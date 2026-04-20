"""Unit tests for speaker_embed operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import wespeaker  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip("wespeaker not available", allow_module_level=True)

from voxkitchen.operators.annotate.speaker_embed import (
    SpeakerEmbedConfig,
    SpeakerEmbedOperator,
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


def test_speaker_embed_is_registered() -> None:
    assert get_operator("speaker_embed") is SpeakerEmbedOperator


@pytest.mark.slow
def test_speaker_embed_wespeaker_extracts_embedding(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    ctx = make_run_context("speaker")
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = SpeakerEmbedConfig(method="wespeaker")
    op = SpeakerEmbedOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert "speaker_embedding" in out_cut.custom
    emb = out_cut.custom["speaker_embedding"]
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert "speaker_embedding_model" in out_cut.custom
