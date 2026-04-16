"""Runtime context passed to operators and executors.

RunContext is a small dataclass containing everything an operator might need
at runtime (paths, identifiers, device assignment) without pulling in any
unpicklable objects like file handles or logger instances. This is crucial
for the multiprocessing-based executors — ``RunContext`` crosses process
boundaries via pickle, so it must stay simple.

Logging is handled through ``logging.getLogger(...)`` inside each worker;
loggers are looked up by name, not passed around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunContext:
    """Pickle-friendly runtime context for a single stage of a pipeline run."""

    work_dir: Path
    pipeline_run_id: str
    stage_index: int
    stage_name: str
    num_gpus: int
    num_cpu_workers: int
    gc_mode: Literal["aggressive", "keep"]
    device: str  # "cpu" or "cuda:0" from the worker's view

    @property
    def stage_dir(self) -> Path:
        """Directory for this stage's outputs: ``work_dir/NN_<name>/``."""
        return self.work_dir / f"{self.stage_index:02d}_{self.stage_name}"

    def with_stage(self, *, stage_index: int, stage_name: str) -> RunContext:
        """Return a copy of this context advanced to a new stage."""
        return replace(self, stage_index=stage_index, stage_name=stage_name)
