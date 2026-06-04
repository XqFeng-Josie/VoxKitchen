"""Coverage gate: every registered operator must have a sweep pipeline yaml."""

from pathlib import Path


def test_every_registered_op_has_a_sweep_pipeline() -> None:
    import voxkitchen.operators  # noqa: F401  # trigger discovery
    from voxkitchen.operators.registry import list_operators

    pipelines_dir = Path(__file__).resolve().parents[3] / "scripts" / "sweep" / "pipelines"
    have_yaml = {p.stem for p in pipelines_dir.glob("*.yaml")}
    # Exclude test-only sentinel operators (names starting with "_") that are
    # registered in test files for executor/GPU testing and don't need yamls.
    registered = {op for op in list_operators() if not op.startswith("_")}

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
