"""Subprocess entry point: run one pipeline stage in this env.

The parent pipeline runner spawns this when a stage's operator lives in a
different Python environment. Protocol:

Inputs (CLI args):
  --op         operator name (registered in this env)
  --config-json  operator config serialized as JSON
  --input      path to input ``cuts.jsonl.gz``
  --output     path to output ``cuts.jsonl.gz``
  --ctx-json   RunContext serialized as JSON (work_dir, stage_index, ...)

Outputs on disk:
  <output>                  the transformed CutSet
  <output>.parent/_errors.jsonl   per-cut errors (if any)
  <output>.parent/_stats.json     wall time, cuts in/out, throughput

Exit codes:
  0  success (output and metadata written)
  1  unrecoverable failure (stderr carries the traceback)
  2  invalid CLI invocation

The parent treats this subprocess as opaque. Everything that needs to
survive the subprocess boundary is written to disk; stderr is streamed
through to the parent for diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext
    from voxkitchen.pipeline.executor import CpuPoolExecutor, GpuPoolExecutor

logger = logging.getLogger("voxkitchen.stage_runner")

_CTX_STR_FIELDS = {"pipeline_run_id", "stage_name", "gc_mode", "device"}
_CTX_INT_FIELDS = {"stage_index", "num_gpus", "num_cpu_workers"}
_CTX_PATH_FIELDS = {"work_dir"}


def _deserialize_ctx(ctx_json: str) -> RunContext:
    """Rebuild a :class:`RunContext` from the JSON passed on the CLI.

    The caller serializes with ``dataclasses.asdict`` (paths turned to str).
    We validate field names/types here because this is the subprocess
    boundary — garbage in should fail loudly, not at operator runtime.
    """
    from voxkitchen.pipeline.context import RunContext

    raw = json.loads(ctx_json)
    known = {f.name for f in fields(RunContext)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown RunContext fields: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _CTX_PATH_FIELDS:
            kwargs[key] = Path(value)
        elif key in _CTX_INT_FIELDS:
            kwargs[key] = int(value)
        elif key in _CTX_STR_FIELDS:
            kwargs[key] = str(value)
        else:
            kwargs[key] = value
    return RunContext(**kwargs)


def run_stage(
    *,
    op_name: str,
    config_json: str,
    input_path: Path,
    output_path: Path,
    ctx_json: str,
) -> int:
    """Execute a single stage end-to-end in this env. Returns exit code."""
    # Importing this populates the registry with operators this env can load.
    import voxkitchen.operators  # noqa: F401  (side effect)
    from voxkitchen.operators.registry import MissingExtrasError, UnknownOperatorError, get_operator
    from voxkitchen.pipeline.executor import CpuPoolExecutor, GpuPoolExecutor
    from voxkitchen.schema.cutset import CutSet
    from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord

    ctx = _deserialize_ctx(ctx_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        op_cls = get_operator(op_name)
    except (UnknownOperatorError, MissingExtrasError) as exc:
        # Unrecoverable: this env doesn't have the op. The parent should
        # not have dispatched here — bug in op_env_map.json or runner.
        print(f"stage_runner: cannot load operator {op_name!r}: {exc}", file=sys.stderr)
        return 1

    try:
        config = op_cls.config_cls.model_validate_json(config_json)
    except Exception as exc:
        print(f"stage_runner: invalid config for {op_name!r}: {exc}", file=sys.stderr)
        return 1

    # Executor selection: same logic as the in-process runner. We re-use
    # Cpu/GpuPoolExecutor here, which internally use multiprocessing.spawn
    # — that's fine, each subprocess-pool worker is in the SAME env as
    # this stage_runner, just forked from it.
    from voxkitchen.pipeline.runner import _gpu_available  # reuse helper

    executor: CpuPoolExecutor | GpuPoolExecutor
    if str(op_cls.device) == "gpu" and ctx.num_gpus > 0 and _gpu_available():
        executor = GpuPoolExecutor(num_gpus=ctx.num_gpus)
    else:
        if str(op_cls.device) == "gpu" and not _gpu_available():
            logger.info("GPU not available, falling back to CPU executor")
        executor = CpuPoolExecutor(num_workers=max(1, ctx.num_cpu_workers))

    try:
        cuts_in = CutSet.from_jsonl_gz(input_path)
    except FileNotFoundError as exc:
        print(f"stage_runner: input manifest not found: {exc}", file=sys.stderr)
        return 1

    n_input = len(cuts_in)
    t_start = time.monotonic()
    try:
        cuts_out = executor.run(op_cls, config, cuts_in, ctx)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    wall_time = time.monotonic() - t_start

    import datetime

    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.datetime.now(tz=datetime.timezone.utc),
        pipeline_run_id=ctx.pipeline_run_id,
        stage_name=ctx.stage_name,
    )
    cuts_out.to_jsonl_gz(output_path, header)

    stats = {
        "stage_name": ctx.stage_name,
        "operator": op_name,
        "wall_time_seconds": round(wall_time, 2),
        "cuts_in": n_input,
        "cuts_out": len(cuts_out),
        "throughput_cuts_per_sec": round(len(cuts_out) / wall_time, 2) if wall_time > 0 else 0.0,
        "env": _current_env_name(),
    }
    (output_path.parent / "_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return 0


def _current_env_name() -> str:
    from voxkitchen.runtime.env_resolver import current_env

    return current_env()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, help="Registered operator name.")
    parser.add_argument("--config-json", required=True, help="Operator config as JSON.")
    parser.add_argument("--input", required=True, type=Path, help="Input cuts.jsonl.gz.")
    parser.add_argument("--output", required=True, type=Path, help="Output cuts.jsonl.gz.")
    parser.add_argument("--ctx-json", required=True, help="RunContext as JSON.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(levelname)s %(message)s",
    )
    return run_stage(
        op_name=args.op,
        config_json=args.config_json,
        input_path=args.input,
        output_path=args.output,
        ctx_json=args.ctx_json,
    )


if __name__ == "__main__":
    sys.exit(main())
