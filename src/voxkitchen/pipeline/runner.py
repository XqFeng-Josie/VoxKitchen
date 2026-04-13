"""The pipeline runner: orchestrates stages end-to-end.

The runner is the only module that knows about all the other pipeline
pieces at once. It:

1. Resolves the ingest source and produces the initial CutSet.
2. Computes the GC plan.
3. Detects resume point (highest contiguously-completed stage).
4. For each remaining stage:
   a. Instantiates the operator.
   b. Picks an executor based on ``operator.device``.
   c. Runs the operator over the CutSet.
   d. Writes output manifest + _SUCCESS marker.
   e. Runs GC for anything the just-finished stage unblocks.
5. Empties trash on success.

All state is on disk under ``work_dir`` so any crash leaves a resumable tree.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from voxkitchen.ingest import get_ingest_source
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import (
    CpuPoolExecutor,
    Executor,
    GpuPoolExecutor,
)
from voxkitchen.pipeline.gc import compute_gc_plan, empty_trash, run_gc
from voxkitchen.pipeline.spec import PipelineSpec
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord

logger = logging.getLogger(__name__)


class StageFailedError(RuntimeError):
    """Raised when a stage cannot be completed."""

    def __init__(self, stage_name: str, cause: BaseException) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"stage {stage_name!r} failed: {cause}")


def _make_executor(device: str, ctx: RunContext) -> Executor:
    if device == "gpu":
        return GpuPoolExecutor(num_gpus=max(1, ctx.num_gpus))
    return CpuPoolExecutor(num_workers=max(1, ctx.num_cpu_workers))


def _write_run_snapshot(work_dir: Path, spec: PipelineSpec, run_id: str) -> None:
    snapshot = {
        "__voxkitchen_snapshot__": True,
        "run_id": run_id,
        "spec": spec.model_dump(mode="json"),
    }
    (work_dir / "run.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")


def run_pipeline(
    spec: PipelineSpec,
    *,
    stop_at: str | None = None,
    keep_intermediates: bool = False,
) -> None:
    """Execute a pipeline end-to-end with resume support.

    ``stop_at`` — if set, stop after this stage name successfully completes.
    ``keep_intermediates`` — override ``spec.gc_mode`` to ``"keep"``.
    """
    work_dir = Path(spec.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gc_mode = "keep" if keep_intermediates else spec.gc_mode

    # Prefer the run_id stored in the spec (set by the loader); fall back to path heuristic.
    run_id = spec.run_id if spec.run_id else _extract_run_id(spec.work_dir)
    _write_run_snapshot(work_dir, spec, run_id)

    base_ctx = RunContext(
        work_dir=work_dir,
        pipeline_run_id=run_id,
        stage_index=0,
        stage_name="ingest",
        num_gpus=spec.num_gpus,
        num_cpu_workers=spec.num_cpu_workers or 1,
        gc_mode=gc_mode,
        device="cpu",
    )

    stage_names = [s.name for s in spec.stages]
    gc_plan = compute_gc_plan(spec)

    # Detect resume point
    last_complete = find_last_completed_stage(work_dir, stage_names)
    start_idx = 0 if last_complete is None else last_complete + 1

    if start_idx == 0:
        # Fresh run — perform ingest
        current_cuts = _run_ingest(spec, base_ctx)
    else:
        # Resume — load the last completed stage's manifest.
        # last_complete is not None here because start_idx > 0 implies last_complete >= 0.
        assert last_complete is not None
        resume_dir = work_dir / stage_dir_name(last_complete, stage_names[last_complete])
        logger.info("resuming from stage %s", resume_dir.name)
        current_cuts = CutSet.from_jsonl_gz(resume_dir / "cuts.jsonl.gz")

    # Execute remaining stages
    for idx in range(start_idx, len(spec.stages)):
        stage = spec.stages[idx]
        stage_ctx = replace(base_ctx, stage_index=idx, stage_name=stage.name)
        stage_dir = stage_ctx.stage_dir
        stage_dir.mkdir(parents=True, exist_ok=True)

        if is_stage_complete(stage_dir):
            logger.info("stage %s already complete, skipping", stage.name)
            current_cuts = CutSet.from_jsonl_gz(stage_dir / "cuts.jsonl.gz")
            continue

        try:
            op_cls = get_operator(stage.op)
            op_cfg = op_cls.config_cls.model_validate(stage.args)
            executor = _make_executor(op_cls.device, stage_ctx)
            current_cuts = executor.run(op_cls, op_cfg, current_cuts, stage_ctx)
        except Exception as exc:
            raise StageFailedError(stage.name, exc) from exc

        # Persist output + marker
        header = HeaderRecord(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(tz=timezone.utc),
            pipeline_run_id=run_id,
            stage_name=stage.name,
        )
        current_cuts.to_jsonl_gz(stage_dir / "cuts.jsonl.gz", header)
        write_success_marker(stage_dir)

        # GC after each stage
        run_gc(
            gc_plan,
            work_dir=work_dir,
            just_completed_idx=idx,
            gc_mode=gc_mode,
            stage_names=stage_names,
        )

        if stop_at == stage.name:
            logger.info("stop_at=%s reached, exiting", stop_at)
            return

    # Generate report (non-critical — catch and log errors)
    try:
        from voxkitchen.viz.report.generator import generate_report

        generate_report(work_dir, pipeline_name=spec.name, run_id=run_id)
        logger.info("report generated: %s/report.html", work_dir)
    except ImportError:
        logger.debug("viz extras not installed; skipping report generation")
    except Exception:
        logger.warning("report generation failed", exc_info=True)

    # Success — empty trash (unless keep mode)
    if gc_mode == "aggressive":
        empty_trash(work_dir)


def _run_ingest(spec: PipelineSpec, ctx: RunContext) -> CutSet:
    source_cls = get_ingest_source(spec.ingest.source)
    args = dict(spec.ingest.args)
    if spec.ingest.recipe is not None:
        args.setdefault("recipe", spec.ingest.recipe)
    config = source_cls.config_cls.model_validate(args)
    source = source_cls(config, ctx)
    return source.run()


def _extract_run_id(work_dir_str: str) -> str:
    """Extract the run id from the work_dir path (for resume logging).

    The work_dir path has ${run_id} already expanded by the loader; we just
    read it back. If no ``run-`` substring exists, fall back to a fresh id.
    """
    for part in Path(work_dir_str).parts:
        if part.startswith("run-"):
            return part
    from voxkitchen.utils.run_id import generate_run_id

    return generate_run_id()
