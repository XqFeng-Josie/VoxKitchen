"""Unit tests for voxkitchen.pipeline.executor.CpuPoolExecutor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.operators.base import Operator, OperatorConfig
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


class BatchOnlyOperator(Operator):
    name = "batch_only_for_test"
    config_cls = OperatorConfig
    parallelizable = False

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet([_cut(f"batch-{len(cuts)}")])


class BatchFailingOperator(Operator):
    name = "batch_failing_for_test"
    config_cls = OperatorConfig
    parallelizable = False

    def process(self, cuts: CutSet) -> CutSet:
        if len(cuts) > 1:
            raise RuntimeError("batch failure")
        return CutSet(list(cuts))


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


def test_cpu_pool_executor_clears_stale_error_file_on_success(tmp_path: Path) -> None:
    cs = CutSet([_cut("c0")])
    ctx = _ctx(tmp_path)
    ctx.stage_dir.mkdir(parents=True)
    errors_path = ctx.stage_dir / "_errors.jsonl"
    errors_path.write_text('{"cut_id":"old"}\n', encoding="utf-8")

    result = CpuPoolExecutor(num_workers=1).run(IdentityOperator, IdentityConfig(), cs, ctx)

    assert [c.id for c in result] == ["c0"]
    assert not errors_path.exists()


def test_cpu_pool_executor_runs_non_parallelizable_operator_once(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(10)])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=4)

    result = executor.run(BatchOnlyOperator, OperatorConfig(), cs, ctx)

    assert [c.id for c in result] == ["batch-10"]


def test_cpu_pool_executor_does_not_cut_fallback_non_parallelizable_operator(
    tmp_path: Path,
) -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=4)

    with pytest.raises(RuntimeError, match="batch failure"):
        executor.run(BatchFailingOperator, OperatorConfig(), cs, ctx)


class FailOnCutOperator(Operator):
    """Raises RuntimeError for any cut whose id contains 'fail'.

    Used to verify the per-cut fallback: the failing cut must be recorded to
    _errors.jsonl while good cuts are returned normally.
    """

    name = "fail_on_cut_for_test"
    config_cls = OperatorConfig
    parallelizable = True

    def process(self, cuts: CutSet) -> CutSet:
        out: list[Cut] = []
        for cut in cuts:
            if "fail" in cut.id:
                raise RuntimeError(f"deliberate failure for cut {cut.id}")
            out.append(cut)
        return CutSet(out)


def test_failing_cut_recorded_to_errors_jsonl(tmp_path: Path) -> None:
    """Executor contract: a cut whose processing raises must be written to
    _errors.jsonl; good cuts must pass through unchanged.

    This is the regression guard that proves removing the operator-level
    'except Exception: return None' actually surfaces errors rather than
    silently dropping cuts.
    """
    import json

    good_cut = _cut("good-1")
    fail_cut = _cut("fail-1")
    cs = CutSet([good_cut, fail_cut])
    ctx = _ctx(tmp_path)
    ctx.stage_dir.mkdir(parents=True, exist_ok=True)

    result = CpuPoolExecutor(num_workers=1).run(FailOnCutOperator, OperatorConfig(), cs, ctx)

    # Good cut must survive
    result_ids = [c.id for c in result]
    assert "good-1" in result_ids, f"good cut must survive; got {result_ids}"
    assert "fail-1" not in result_ids, f"failing cut must be dropped; got {result_ids}"

    # Error must be recorded
    errors_path = ctx.stage_dir / "_errors.jsonl"
    assert errors_path.exists(), "_errors.jsonl must be written when a cut fails"
    records = [json.loads(line) for line in errors_path.read_text().splitlines()]
    assert len(records) == 1, f"expected 1 error record, got {len(records)}"
    assert records[0]["cut_id"] == "fail-1"
    assert records[0]["stage"] == ctx.stage_name
    assert "deliberate failure" in records[0]["error"]
