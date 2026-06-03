"""Report rendering for the operator sweep (stub — populated in Task 11)."""

from pathlib import Path


def write_report(*, records: list, path: Path) -> None:
    """Stub — final implementation in Task 11."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Operator sweep — stub report\n", "(report renderer arrives in Task 11)\n"]
    for r in records:
        lines.append(f"- {r.op} ({r.image}): {r.verdict} — {r.message}\n")
    path.write_text("".join(lines))
