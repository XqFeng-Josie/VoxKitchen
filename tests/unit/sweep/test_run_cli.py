"""Tests for the sweep driver's CLI behaviour (without real docker calls)."""

from pathlib import Path


def test_run_cli_help_lists_main_flags() -> None:
    """`python scripts/sweep/run.py --help` shows --op, --image, --no-pull,
    --cleanup-each, --setup, --report-only flags."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/sweep/run.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode == 0, result.stderr
    for flag in ["--op", "--image", "--no-pull", "--cleanup-each", "--setup", "--report-only"]:
        assert flag in result.stdout, f"missing flag {flag} in --help output"


def test_run_cli_missing_yaml_for_op_exits_nonzero() -> None:
    """Asking for an op without a pipeline yaml exits non-zero with a clear error."""
    import subprocess

    result = subprocess.run(
        ["python", "scripts/sweep/run.py", "--op", "no_such_op_no_yaml", "--no-pull"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.returncode != 0
    assert "pipeline" in result.stdout.lower() or "pipeline" in result.stderr.lower()
