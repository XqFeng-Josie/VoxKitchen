"""Aggressive garbage collection for materialized audio in ``derived/`` dirs.

The plan is computed once at pipeline startup by scanning the stage list for
operators with ``produces_audio = True`` and finding, for each such producer,
the last downstream stage whose operator has ``reads_audio_bytes = True``.
That downstream stage is the ``last_consumer`` — once it completes, the
producer's derived files are no longer needed and can be moved to
``derived_trash/``.

``derived_trash/`` is emptied only after the entire pipeline completes
successfully. A crash during the run leaves trash in place for diagnostics.

The final stage is never GC'd — its output is the user-facing artifact.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.checkpoint import stage_dir_name
from voxkitchen.pipeline.spec import PipelineSpec


@dataclass
class GcPlan:
    """Static analysis result: producer stage index → last consumer stage index."""

    last_consumer: dict[int, int] = field(default_factory=dict)


def compute_gc_plan(spec: PipelineSpec) -> GcPlan:
    """Scan a PipelineSpec and produce a GcPlan.

    For each stage whose operator ``produces_audio`` (and is not the final
    stage), find the last downstream stage whose operator ``reads_audio_bytes``.
    Record the pair as ``(producer_idx, last_consumer_idx)`` in the plan.
    """
    last_consumer: dict[int, int] = {}
    num_stages = len(spec.stages)

    for i, stage in enumerate(spec.stages):
        op_cls = get_operator(stage.op)
        if not op_cls.produces_audio:
            continue
        if i == num_stages - 1:
            # Final stage is the artifact; never GC'd
            continue

        # Find the last downstream consumer that reads audio bytes.
        # Stop scanning at the next producer (inclusive) — a downstream stage
        # that also produces_audio re-materialises the audio, so subsequent
        # stages consume *its* derived files, not ours.
        consumer_idx: int | None = None
        for j in range(i + 1, num_stages):
            downstream_op = get_operator(spec.stages[j].op)
            if downstream_op.reads_audio_bytes:
                consumer_idx = j
            if downstream_op.produces_audio:
                # This stage re-materialises; no further stage uses our audio
                break
        if consumer_idx is not None:
            last_consumer[i] = consumer_idx

    return GcPlan(last_consumer=last_consumer)


def run_gc(
    plan: GcPlan,
    *,
    work_dir: Path,
    just_completed_idx: int,
    gc_mode: Literal["aggressive", "keep"],
    stage_names: list[str] | None = None,
) -> None:
    """Move any now-unneeded ``derived/`` directories to ``derived_trash/``.

    ``stage_names``, if provided, is used to resolve producer indices to
    on-disk directory names. When omitted, callers are expected to pass it
    because directory names embed the stage name, not just the index.
    """
    if gc_mode == "keep":
        return
    if stage_names is None:
        stage_names = []

    trash_root = work_dir / "derived_trash"

    for producer_idx, consumer_idx in plan.last_consumer.items():
        if consumer_idx != just_completed_idx:
            continue
        if producer_idx >= len(stage_names):
            continue
        producer_name = stage_names[producer_idx]
        producer_dir = work_dir / stage_dir_name(producer_idx, producer_name)
        derived = producer_dir / "derived"
        if not derived.exists():
            continue
        target = trash_root / stage_dir_name(producer_idx, producer_name) / "derived"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(derived), str(target))


def empty_trash(work_dir: Path) -> None:
    """Permanently delete the ``derived_trash/`` directory. Safe to call twice."""
    trash = work_dir / "derived_trash"
    if trash.exists():
        shutil.rmtree(trash)
