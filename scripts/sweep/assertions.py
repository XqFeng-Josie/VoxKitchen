"""Per-operator assertion functions (stub — populated in Task 3)."""

from collections.abc import Callable
from pathlib import Path


def default_smoke_assertion(work_dir: Path, _log: str) -> tuple[bool, str]:
    """Trivial fallback — every op-under-test PASSes if the pipeline exited 0.

    Replaced by the real implementation in Task 3.
    """
    return True, "stub (Task 3 will read final cuts)"


ASSERTIONS: dict[str, Callable[[Path, str], tuple[bool, str]]] = {}
