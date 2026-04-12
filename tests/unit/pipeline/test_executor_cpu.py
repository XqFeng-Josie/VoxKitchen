"""Unit tests for voxkitchen.pipeline.executor.CpuPoolExecutor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.noop.identity import IdentityConfig, IdentityOperator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import CpuPoolExecutor
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-a1b2c3",
        stage_index=1,
        stage_name="identity",
        num_gpus=0,
        num_cpu_workers=2,
        gc_mode="aggressive",
        device="cpu",
    )


def test_cpu_pool_executor_preserves_all_cuts(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(10)])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=2)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert sorted(c.id for c in result) == sorted(c.id for c in cs)


def test_cpu_pool_executor_with_single_worker(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(3)])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=1)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert [c.id for c in result] == [c.id for c in cs]


def test_cpu_pool_executor_handles_empty_cutset(tmp_path: Path) -> None:
    cs = CutSet([])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=2)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert len(result) == 0
