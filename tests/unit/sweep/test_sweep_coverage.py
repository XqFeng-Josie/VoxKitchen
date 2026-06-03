"""Coverage gate: every registered operator must have a sweep pipeline yaml.

This test is xfailed until Tasks 5-10 author all 52 sweep pipelines. The
xfail comes off in Task 10. Until then it serves as a reminder of the
authoring debt - `pytest tests/unit/sweep/test_sweep_coverage.py -v`
prints the current set of missing yamls so the next Task knows what's
left.
"""

from pathlib import Path

import pytest


@pytest.mark.xfail(
    reason=(
        "Coverage gate fails until Tasks 5-10 author all 52 sweep pipelines. "
        "Remove this xfail when Task 10 commits the last batch."
    ),
    strict=False,
)
def test_every_registered_op_has_a_sweep_pipeline() -> None:
    import voxkitchen.operators  # noqa: F401  # trigger discovery
    from voxkitchen.operators.registry import list_operators

    pipelines_dir = Path(__file__).resolve().parents[3] / "scripts" / "sweep" / "pipelines"
    have_yaml = {p.stem for p in pipelines_dir.glob("*.yaml")}
    registered = set(list_operators())

    missing = sorted(registered - have_yaml)
    extra = sorted(have_yaml - registered)

    issues = []
    if missing:
        issues.append(
            f"{len(missing)} registered ops missing a sweep pipeline "
            f"(scripts/sweep/pipelines/<op>.yaml): {missing}"
        )
    if extra:
        issues.append(
            f"{len(extra)} sweep pipelines for ops not in the registry (stale yamls?): {extra}"
        )

    assert not issues, "\n".join(issues)
