"""Unit tests for pack_jsonl operator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.pack.pack_jsonl import (
    PackJsonlConfig,
    PackJsonlOperator,
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


def test_pack_jsonl_is_registered() -> None:
    assert get_operator("pack_jsonl") is PackJsonlOperator


def test_pack_jsonl_produces_no_audio() -> None:
    assert PackJsonlOperator.produces_audio is False


def test_pack_jsonl_writes_output_file(
    mono_wav_16k: Path, tmp_path: Path, make_run_context
) -> None:
    """pack_jsonl writes a manifest.jsonl file with one line per cut."""
    ctx = make_run_context("pack")
    cut = _cut_from_path(mono_wav_16k)
    cs = CutSet([cut])

    output_path = tmp_path / "manifest.jsonl"
    config = PackJsonlConfig(output_path=str(output_path))
    op = PackJsonlOperator(config, ctx=ctx)
    op.process(cs)

    assert output_path.exists()
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert "id" in row
    assert "duration" in row
    assert row["duration"] > 0


def test_pack_jsonl_returns_all_cuts(mono_wav_16k: Path, tmp_path: Path, make_run_context) -> None:
    """pack_jsonl returns the same number of cuts it received."""
    ctx = make_run_context("pack")
    cut = _cut_from_path(mono_wav_16k)
    cs = CutSet([cut])

    output_path = tmp_path / "out.jsonl"
    config = PackJsonlConfig(output_path=str(output_path))
    op = PackJsonlOperator(config, ctx=ctx)
    result = list(op.process(cs))

    assert len(result) == 1
