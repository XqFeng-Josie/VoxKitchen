"""Unit tests for forced_align operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import qwen_asr  # noqa: F401
except ImportError:
    pytest.skip("qwen-asr not available", allow_module_level=True)

from voxkitchen.operators.annotate.forced_align import (
    ForcedAlignConfig,
    ForcedAlignOperator,
)
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import recording_from_file


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="align",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _cut_no_text(audio_path: Path) -> Cut:
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


def _cut_with_text(audio_path: Path, text: str) -> Cut:
    rec = recording_from_file(audio_path)
    return Cut(
        id=f"cut-{rec.id}",
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
        recording=rec,
        supervisions=[
            Supervision(
                id=f"sup-{rec.id}",
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                text=text,
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_forced_align_is_registered() -> None:
    assert get_operator("forced_align") is ForcedAlignOperator


def test_forced_align_does_not_produce_audio() -> None:
    assert ForcedAlignOperator.produces_audio is False


def test_forced_align_skips_cuts_without_text(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cut = _cut_no_text(mono_wav_16k)
    cs = CutSet([cut])
    config = ForcedAlignConfig()
    op = ForcedAlignOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()
    out_cut = next(iter(result))
    assert "word_alignments" not in out_cut.custom
