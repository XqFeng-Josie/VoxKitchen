"""Unit tests for cer_wer operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.quality.cer_wer import (
    CerWerConfig,
    CerWerOperator,
)
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="cer",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _cut(cid: str, hyp_text: str, ref_text: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[
            Supervision(
                id=f"sup-{cid}",
                recording_id="rec-1",
                start=0.0,
                duration=1.0,
                text=hyp_text,
            )
        ],
        custom={"reference_text": ref_text},
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_cer_wer_is_registered() -> None:
    assert get_operator("cer_wer") is CerWerOperator


def test_cer_wer_identical_text(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut("c0", "hello world", "hello world")])
    config = CerWerConfig()
    op = CerWerOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.metrics["cer"] == 0.0
    assert out.metrics["wer"] == 0.0


def test_cer_wer_completely_wrong(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut("c0", "xyz", "abc")])
    config = CerWerConfig()
    op = CerWerOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert out.metrics["cer"] == pytest.approx(1.0, abs=0.01)
    assert out.metrics["wer"] == pytest.approx(1.0, abs=0.01)


def test_cer_wer_partial_match(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut("c0", "hello wrld", "hello world")])
    config = CerWerConfig()
    op = CerWerOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert 0.0 < out.metrics["cer"] < 0.5
    assert 0.0 < out.metrics["wer"] <= 1.0


def test_cer_wer_no_reference_skips(tmp_path: Path) -> None:
    cut = Cut(
        id="c0",
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[
            Supervision(
                id="sup-c0",
                recording_id="rec-1",
                start=0.0,
                duration=1.0,
                text="hello",
            )
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )
    ctx = _ctx(tmp_path)
    cs = CutSet([cut])
    config = CerWerConfig()
    op = CerWerOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out = next(iter(result))
    assert "cer" not in out.metrics
