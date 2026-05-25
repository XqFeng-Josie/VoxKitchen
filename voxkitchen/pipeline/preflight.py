"""Static pre-flight validation of pipeline stage chains (Workstream A).

Walks stages forward over a set of "available field tokens" (see the field
vocabulary in the plan/spec) and reports broken chains BEFORE execution.
Pure, dependency-free set logic — no type checking, no audio, no models.
"""

from __future__ import annotations


def _namespace(token: str) -> str | None:
    """Return the namespace prefix for a wildcard token like 'metrics.*'."""
    return token[:-2] if token.endswith(".*") else None


def is_satisfied(required: str, available: set[str]) -> bool:
    """True if ``required`` is met by ``available`` (exact or namespace match)."""
    if required in available:
        return True
    # required is specific (e.g. metrics.snr); a wildcard 'metrics.*' satisfies it
    if "." in required:
        prefix = required.rsplit(".", 1)[0]
        if f"{prefix}.*" in available:
            return True
    # required is itself a wildcard 'metrics.*'; any 'metrics.<k>' satisfies it
    ns = _namespace(required)
    if ns is not None:
        return any(a == ns or a.startswith(f"{ns}.") for a in available)
    return False


def apply_writes(available: set[str], writes: list[str]) -> set[str]:
    """Return a new available set with ``writes`` added."""
    return available | set(writes)


def apply_clears(available: set[str], clears: list[str]) -> set[str]:
    """Return a new available set with ``clears`` removed.

    Clearing a wildcard 'metrics.*' removes every 'metrics.<k>' token.
    """
    out = set(available)
    for token in clears:
        ns = _namespace(token)
        if ns is not None:
            out = {a for a in out if not (a == ns or a.startswith(f"{ns}."))}
        else:
            out.discard(token)
    return out
