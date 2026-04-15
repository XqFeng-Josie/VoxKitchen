"""Executors run a single Operator over a CutSet.

Two implementations ship with Plan 2:

- ``CpuPoolExecutor`` — splits the CutSet into N shards and runs an Operator
  worker in a ``multiprocessing.Pool`` (spawn context) over each shard.
- ``GpuPoolExecutor`` — added in Task 11. Spawns N subprocesses each pinned to
  one GPU via ``CUDA_VISIBLE_DEVICES`` before importing torch.

Both executors share the same Protocol. The runner picks between them based
on ``Operator.device``.
"""

from __future__ import annotations

import json as _json
import logging
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any, Protocol

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cutset import CutSet

logger = logging.getLogger(__name__)


class Executor(Protocol):
    """An Executor knows how to run one Operator over a CutSet."""

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet: ...


def _append_errors(errors_path: Path, errors: list[dict[str, str]]) -> None:
    """Write error records to _errors.jsonl (one JSON object per line).

    Uses write mode (not append) so that re-running a stage replaces stale
    errors from a previous attempt rather than accumulating them.
    """
    if not errors:
        return
    with errors_path.open("w", encoding="utf-8") as f:
        for err in errors:
            f.write(_json.dumps(err, ensure_ascii=False) + "\n")


def _cpu_worker(
    op_cls: type[Operator],
    config_json: str,
    ctx: RunContext,
    cuts_list: list[Any],
    show_progress: bool = False,
) -> tuple[list[Any], list[dict[str, str]]]:
    """Instantiate op, call setup/process/teardown, return processed cuts.

    Config is passed as JSON (not the Pydantic instance) because some
    Pydantic models pickle awkwardly across spawn boundaries; JSON is safe.
    The operator reconstructs its config from JSON inside the worker.
    Cuts are passed as a list of Cut instances (Pydantic v2 pickles cleanly).

    Returns (good_cuts, error_records) so that individual cut failures
    don't crash the entire shard.
    """
    config = op_cls.config_cls.model_validate_json(config_json)
    op = op_cls(config, ctx)
    op.setup()
    try:
        input_cuts = CutSet(cuts_list)
        if show_progress:
            input_cuts = input_cuts.with_progress(desc=ctx.stage_name)
        output_cuts = op.process(input_cuts)
        return list(output_cuts), []
    except Exception:
        # Per-cut fallback: retry one-by-one so only bad cuts are skipped
        good: list[Any] = []
        errors: list[dict[str, str]] = []
        for cut in cuts_list:
            try:
                result = op.process(CutSet([cut]))
                good.extend(list(result))
            except Exception:
                tb_lines = traceback.format_exc().strip().split("\n")
                short = tb_lines[-1]
                errors.append(
                    {
                        "cut_id": cut.id,
                        "stage": ctx.stage_name,
                        "error": short,
                        "traceback": "\n".join(tb_lines[-4:]),
                    }
                )
                logger.warning("cut %s failed: %s", cut.id, short)
        return good, errors
    finally:
        op.teardown()


class CpuPoolExecutor:
    """Shard a CutSet across a multiprocessing.Pool of CPU workers."""

    def __init__(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.num_workers = num_workers

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet:
        if len(cuts) == 0:
            return CutSet([])

        effective_workers = min(self.num_workers, len(cuts))
        shards = cuts.split(effective_workers)
        config_json = config.model_dump_json()

        if effective_workers == 1:
            good, errors = _cpu_worker(
                op_cls,
                config_json,
                ctx,
                list(shards[0]),
                show_progress=True,
            )
            _append_errors(ctx.stage_dir / "_errors.jsonl", errors)
            return CutSet(good)

        tasks = [(op_cls, config_json, ctx, list(shard)) for shard in shards]

        ctx_mp = mp.get_context("spawn")
        with ctx_mp.Pool(effective_workers) as pool:
            results = pool.starmap(_cpu_worker, tasks)

        merged: list[Any] = []
        all_errors: list[dict[str, str]] = []
        for good, errors in results:
            merged.extend(good)
            all_errors.extend(errors)
        _append_errors(ctx.stage_dir / "_errors.jsonl", all_errors)
        return CutSet(merged)


def _gpu_worker(
    gpu_id: int,
    op_cls: type[Operator],
    config_json: str,
    ctx: RunContext,
    cuts_list: list[Any],
) -> tuple[list[Any], list[dict[str, str]]]:
    """GPU worker entry point.

    Sets ``CUDA_VISIBLE_DEVICES`` to its assigned id BEFORE torch has any
    chance to be imported. The operator's ``setup()`` (called after this)
    can safely ``import torch`` and see only the intended GPU as ``cuda:0``.
    """
    import os as _os

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from dataclasses import replace as _replace

    worker_ctx = _replace(ctx, device="cuda:0")

    config = op_cls.config_cls.model_validate_json(config_json)
    op = op_cls(config, worker_ctx)
    op.setup()
    try:
        input_cuts = CutSet(cuts_list)
        output_cuts = op.process(input_cuts)
        return list(output_cuts), []
    except Exception:
        good: list[Any] = []
        errors: list[dict[str, str]] = []
        for cut in cuts_list:
            try:
                result = op.process(CutSet([cut]))
                good.extend(list(result))
            except Exception:
                tb_lines = traceback.format_exc().strip().split("\n")
                short = tb_lines[-1]
                errors.append(
                    {
                        "cut_id": cut.id,
                        "stage": worker_ctx.stage_name,
                        "error": short,
                        "traceback": "\n".join(tb_lines[-4:]),
                    }
                )
                logger.warning("cut %s failed on GPU %d: %s", cut.id, gpu_id, short)
        return good, errors
    finally:
        op.teardown()


class GpuPoolExecutor:
    """Shard a CutSet across N GPU-pinned subprocesses.

    Each worker is a fresh ``spawn`` process with ``CUDA_VISIBLE_DEVICES=i``
    set before any torch import. Operators that need GPUs use ``cuda:0`` in
    their ``setup()`` — that maps to the correct physical GPU via the env var.
    """

    def __init__(self, num_gpus: int) -> None:
        if num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
        self.num_gpus = num_gpus

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet:
        if len(cuts) == 0:
            return CutSet([])

        effective_workers = min(self.num_gpus, len(cuts))
        shards = cuts.split(effective_workers)
        config_json = config.model_dump_json()

        tasks = [
            (gpu_id, op_cls, config_json, ctx, list(shard)) for gpu_id, shard in enumerate(shards)
        ]

        ctx_mp = mp.get_context("spawn")
        with ctx_mp.Pool(effective_workers) as pool:
            results = pool.starmap(_gpu_worker, tasks)

        merged: list[Any] = []
        all_errors: list[dict[str, str]] = []
        for good, errors in results:
            merged.extend(good)
            all_errors.extend(errors)
        _append_errors(ctx.stage_dir / "_errors.jsonl", all_errors)
        return CutSet(merged)
