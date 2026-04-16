"""Resume / checkpoint helpers.

A stage is considered complete iff both ``cuts.jsonl.gz`` and ``_SUCCESS``
exist in its directory. The ``_SUCCESS`` marker is written by the runner
only after the manifest has been fully flushed to disk — a crash during
write leaves the manifest present but the marker absent, and the runner
will re-run the stage on the next attempt.
"""

from __future__ import annotations

from pathlib import Path


def stage_dir_name(index: int, name: str) -> str:
    """Return the canonical on-disk directory name: ``NN_<name>``."""
    return f"{index:02d}_{name}"


def write_success_marker(stage_dir: Path) -> None:
    """Create the empty ``_SUCCESS`` marker inside a stage directory."""
    (stage_dir / "_SUCCESS").touch()


def is_stage_complete(stage_dir: Path) -> bool:
    """Return True iff both the manifest and the success marker exist."""
    if not stage_dir.is_dir():
        return False
    return (stage_dir / "cuts.jsonl.gz").exists() and (stage_dir / "_SUCCESS").exists()


def find_last_completed_stage(work_dir: Path, stage_names: list[str]) -> int | None:
    """Return the index of the last contiguously-completed stage, or None.

    Scans stages in order and stops at the first gap — resume only restarts
    from the last known-good point, never "skips" an incomplete stage.
    """
    last_complete: int | None = None
    for i, name in enumerate(stage_names):
        stage_dir = work_dir / stage_dir_name(i, name)
        if is_stage_complete(stage_dir):
            last_complete = i
        else:
            break
    return last_complete
