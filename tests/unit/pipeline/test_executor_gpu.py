"""Unit tests for voxkitchen.pipeline.executor.GpuPoolExecutor.

The tests don't require a real GPU. ``IdentityOperator`` never touches torch,
so the worker process runs happily on CPU-only CI runners. What we verify is
the orchestration: shards are processed by the right number of workers, all
cuts are recovered in the output, and CUDA_VISIBLE_DEVICES is set per worker.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import GpuPoolExecutor
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


class _CudaSentinelConfig(OperatorConfig):
    pass


@register_operator
class _CudaSentinelOperator(Operator):
    """Stamps each Cut's custom dict with the CUDA_VISIBLE_DEVICES it saw."""

    name = "_test_cuda_sentinel"
    config_cls = _CudaSentinelConfig
    device = "gpu"

    def process(self, cuts: CutSet) -> CutSet:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
        out = []
        for c in cuts:
            out.append(c.model_copy(update={"custom": {"cvd": cvd}}))
        return CutSet(out)


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
        stage_name="gpu_stage",
        num_gpus=2,
        num_cpu_workers=0,
        gc_mode="aggressive",
        device="cuda:0",
    )


def test_gpu_pool_executor_recovers_all_cuts(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(6)])
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    assert sorted(c.id for c in result) == sorted(c.id for c in cs)


def test_gpu_pool_executor_sets_cuda_visible_devices_per_worker(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(4)])
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    # Each Cut should carry the CUDA_VISIBLE_DEVICES its worker saw.
    seen = {c.custom["cvd"] for c in result}
    # Should contain "0" and "1" at least (shards may be uneven but both GPUs used
    # since we have 4 cuts and 2 workers)
    assert seen == {"0", "1"}


def test_gpu_pool_executor_empty_cutset(tmp_path: Path) -> None:
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), CutSet([]), _ctx(tmp_path))
    assert len(result) == 0


def test_gpu_pool_executor_uses_single_gpu_when_fewer_cuts(tmp_path: Path) -> None:
    """With 1 cut and num_gpus=4, only 1 worker should be spawned."""
    cs = CutSet([_cut("only")])
    executor = GpuPoolExecutor(num_gpus=4)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    assert len(result) == 1
    only_cut = next(iter(result))
    assert only_cut.custom["cvd"] == "0"
