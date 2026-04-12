"""Unit tests for voxkitchen.pipeline.context.RunContext."""

from __future__ import annotations

import pickle
from pathlib import Path

from voxkitchen.pipeline.context import RunContext


def _ctx(work_dir: Path) -> RunContext:
    return RunContext(
        work_dir=work_dir,
        pipeline_run_id="run-abcd",
        stage_index=2,
        stage_name="vad",
        num_gpus=4,
        num_cpu_workers=8,
        gc_mode="aggressive",
        device="cpu",
    )


def test_run_context_stage_dir_uses_zero_padded_index(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.stage_dir == tmp_path / "02_vad"


def test_run_context_with_stage_returns_new_instance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    new_ctx = ctx.with_stage(stage_index=5, stage_name="asr")
    assert new_ctx.stage_index == 5
    assert new_ctx.stage_name == "asr"
    assert ctx.stage_index == 2
    assert ctx.stage_name == "vad"
    assert new_ctx.pipeline_run_id == ctx.pipeline_run_id
    assert new_ctx.num_gpus == ctx.num_gpus


def test_run_context_is_picklable(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    blob = pickle.dumps(ctx)
    restored = pickle.loads(blob)
    assert restored == ctx


def test_ingest_dir_is_stage_zero(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ingest_ctx = ctx.with_stage(stage_index=0, stage_name="ingest")
    assert ingest_ctx.stage_dir == tmp_path / "00_ingest"
