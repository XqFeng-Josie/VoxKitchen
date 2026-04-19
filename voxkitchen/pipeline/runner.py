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

import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from voxkitchen.ingest import get_ingest_source
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
)
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
from voxkitchen.pipeline.spec import PipelineSpec, StageSpec
from voxkitchen.runtime.dispatch import StageDispatchFailure, dispatch_stage_to_env
from voxkitchen.runtime.env_resolver import current_env, resolve_env
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord

logger = logging.getLogger(__name__)


class StageFailedError(RuntimeError):
    """Raised when a stage cannot be completed."""

    def __init__(self, stage_name: str, cause: BaseException) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"stage {stage_name!r} failed: {cause}")


def _gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _make_executor(device: str, ctx: RunContext) -> Executor:
    if device == "gpu" and ctx.num_gpus > 0 and _gpu_available():
        return GpuPoolExecutor(num_gpus=ctx.num_gpus)
    if device == "gpu" and not _gpu_available():
        logger.info("GPU not available, falling back to CPU executor")
    return CpuPoolExecutor(num_workers=max(1, ctx.num_cpu_workers))


def _run_stage_in_process(
    *,
    stage: StageSpec,
    cuts: CutSet,
    ctx: RunContext,
    run_id: str,
) -> CutSet:
    """Classic in-process execution: load op, validate, run executor, persist.

    Writes the stage output manifest and `_stats.json`. The caller is
    responsible for the `_SUCCESS` marker and GC.
    """
    stage_dir = ctx.stage_dir
    n_input = len(cuts)
    t_start = time.monotonic()
    op_cls = get_operator(stage.op)
    op_cfg = op_cls.config_cls.model_validate(stage.args)
    executor = _make_executor(op_cls.device, ctx)
    result = executor.run(op_cls, op_cfg, cuts, ctx)
    wall_time = time.monotonic() - t_start

    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(tz=timezone.utc),
        pipeline_run_id=run_id,
        stage_name=stage.name,
    )
    result.to_jsonl_gz(stage_dir / "cuts.jsonl.gz", header)
    _write_stage_stats(
        stage_dir=stage_dir,
        stage_name=stage.name,
        operator=stage.op,
        wall_time=wall_time,
        cuts_in=n_input,
        cuts_out=len(result),
    )
    return result


def _run_stage_in_subprocess(
    *,
    stage: StageSpec,
    cuts: CutSet,
    ctx: RunContext,
    target_env: str,
    idx: int,
    stage_names: list[str],
    work_dir: Path,
    run_id: str,
) -> CutSet:
    """Cross-env execution: write input to disk, spawn stage_runner, read output.

    The subprocess itself writes ``cuts.jsonl.gz`` and ``_stats.json`` to
    ``ctx.stage_dir``. We only need to (a) make sure the input is on disk,
    (b) dispatch, (c) reload the output into memory for the next stage.

    Ingest output does not land on disk automatically; when dispatching
    stage 0 cross-env we materialize it at ``ctx.stage_dir/_input.jsonl.gz``.
    Later stages use the previous stage's ``cuts.jsonl.gz`` directly —
    that file is already persisted by whichever executor produced it.
    """
    stage_dir = ctx.stage_dir
    output_path = stage_dir / "cuts.jsonl.gz"

    # Locate (or create) the input manifest the subprocess should read.
    if idx == 0:
        input_path = stage_dir / "_input.jsonl.gz"
    else:
        prev_dir = work_dir / stage_dir_name(idx - 1, stage_names[idx - 1])
        input_path = prev_dir / "cuts.jsonl.gz"

    if not input_path.exists():
        header = HeaderRecord(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(tz=timezone.utc),
            pipeline_run_id=run_id,
            stage_name=f"_input:{stage.name}",
        )
        cuts.to_jsonl_gz(input_path, header)

    try:
        dispatch_stage_to_env(
            target_env=target_env,
            op_name=stage.op,
            config_args=dict(stage.args),
            input_path=input_path,
            output_path=output_path,
            ctx=ctx,
        )
    except StageDispatchFailure as exc:
        raise StageFailedError(stage.name, exc) from exc

    if not output_path.exists():
        raise StageFailedError(
            stage.name,
            FileNotFoundError(
                f"subprocess for {stage.op!r} in env {target_env!r} exited 0 "
                f"but did not produce {output_path}"
            ),
        )
    return CutSet.from_jsonl_gz(output_path)


def _write_stage_stats(
    *,
    stage_dir: Path,
    stage_name: str,
    operator: str,
    wall_time: float,
    cuts_in: int,
    cuts_out: int,
) -> None:
    """Write per-stage execution statistics to _stats.json."""
    throughput = cuts_out / wall_time if wall_time > 0 else 0.0
    stats = {
        "stage_name": stage_name,
        "operator": operator,
        "wall_time_seconds": round(wall_time, 2),
        "cuts_in": cuts_in,
        "cuts_out": cuts_out,
        "throughput_cuts_per_sec": round(throughput, 2),
    }
    (stage_dir / "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


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
    resume_from: str | None = None,
    keep_intermediates: bool = False,
) -> None:
    """Execute a pipeline end-to-end with resume support.

    ``stop_at`` — if set, stop after this stage name successfully completes.
    ``resume_from`` — if set, resume from this stage name (must have a prior
        stage completed, or start from ingest if resuming from stage 0).
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
    if resume_from is not None:
        if resume_from not in stage_names:
            raise StageFailedError(
                resume_from, ValueError(f"unknown stage {resume_from!r}, available: {stage_names}")
            )
        start_idx = stage_names.index(resume_from)
        logger.info("--resume-from %s → starting at stage index %d", resume_from, start_idx)
    else:
        last_complete = find_last_completed_stage(work_dir, stage_names)
        start_idx = 0 if last_complete is None else last_complete + 1

    if start_idx == 0:
        # Fresh run — perform ingest
        current_cuts = _run_ingest(spec, base_ctx)
    else:
        # Resume — load the manifest from the stage just before start_idx.
        prev_idx = start_idx - 1
        prev_dir = work_dir / stage_dir_name(prev_idx, stage_names[prev_idx])
        prev_manifest = prev_dir / "cuts.jsonl.gz"
        if not prev_manifest.exists():
            raise StageFailedError(
                stage_names[start_idx],
                FileNotFoundError(
                    f"cannot resume: prior stage {stage_names[prev_idx]!r} "
                    f"has no manifest at {prev_manifest}"
                ),
            )
        logger.info("resuming from stage %s", prev_dir.name)
        current_cuts = CutSet.from_jsonl_gz(prev_manifest)

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

        n_input = len(current_cuts)
        try:
            target_env = resolve_env(stage.op)
        except Exception as exc:
            raise StageFailedError(stage.name, exc) from exc
        here = current_env()
        logger.info(
            "stage [%d/%d] %s  (%s, %d cuts in, env=%s%s)",
            idx + 1,
            len(spec.stages),
            stage.name,
            stage.op,
            n_input,
            target_env,
            "" if target_env == here else f" ← dispatched from {here}",
        )

        t_start = time.monotonic()
        try:
            if target_env == here:
                current_cuts = _run_stage_in_process(
                    stage=stage,
                    cuts=current_cuts,
                    ctx=stage_ctx,
                    run_id=run_id,
                )
            else:
                current_cuts = _run_stage_in_subprocess(
                    stage=stage,
                    cuts=current_cuts,
                    ctx=stage_ctx,
                    target_env=target_env,
                    idx=idx,
                    stage_names=stage_names,
                    work_dir=work_dir,
                    run_id=run_id,
                )
        except StageFailedError:
            raise
        except Exception as exc:
            raise StageFailedError(stage.name, exc) from exc
        wall_time = time.monotonic() - t_start

        logger.info(
            "stage [%d/%d] %s  done → %d cuts out (%.1fs)",
            idx + 1,
            len(spec.stages),
            stage.name,
            len(current_cuts),
            wall_time,
        )
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
