"""Tests for `vkit doctor`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.runtime import env_resolver


def test_doctor_single_env_no_expect_lists_operators() -> None:
    """In dev mode (no /opt/voxkitchen/envs), doctor reports the current env only."""
    runner = CliRunner()
    with patch.object(env_resolver, "ENVS_DIR", Path("/nonexistent")):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0


def test_doctor_expect_core_succeeds_in_dev_env() -> None:
    """All core operators register in the dev env, so --expect core should pass."""
    runner = CliRunner()
    with patch.object(env_resolver, "ENVS_DIR", Path("/nonexistent")):
        result = runner.invoke(app, ["doctor", "--expect", "core"])
    assert result.exit_code == 0


def test_doctor_json_stdout_is_pure_json(tmp_path: Path) -> None:
    """--json must emit parseable JSON on stdout; rich table goes to stderr."""
    # Use a real subprocess to get true stdout/stderr separation.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    result = subprocess.run(
        [sys.executable, "-m", "voxkitchen.cli.main", "doctor", "--json", "--expect", "core"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["image_kind"] == "core"
    assert "available" in parsed
    assert "missing" in parsed


def _make_fake_envs(tmp_path: Path, env_names: list[str]) -> Path:
    """Create a directory layout that mimics /opt/voxkitchen/envs with a
    working Python interpreter symlinked into each named subdir."""
    envs_dir = tmp_path / "envs"
    for name in env_names:
        bin_dir = envs_dir / name / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "python").symlink_to(sys.executable)
    return envs_dir


def test_doctor_aggregates_multi_env_layout(tmp_path: Path) -> None:
    """When >1 envs are present and --expect is absent, doctor aggregates them."""
    envs_dir = _make_fake_envs(tmp_path, ["core", "asr", "tts"])
    with patch.object(env_resolver, "ENVS_DIR", envs_dir):
        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    # Output format is rich's table on stderr, but CliRunner merges — just
    # ensure each env name appears somewhere.
    for env in ("core", "asr", "tts"):
        assert env in result.output


def test_doctor_expect_slim_aliases_to_core() -> None:
    """`--expect slim` (the image tag) should be accepted as an alias for
    the internal operator group `core` so users don't need to know about
    the tag-vs-group naming distinction."""
    runner = CliRunner()
    with patch.object(env_resolver, "ENVS_DIR", Path("/nonexistent")):
        result = runner.invoke(app, ["doctor", "--expect", "slim"])
    # Should be valid; in dev env all core ops are present so exit 0.
    assert result.exit_code == 0, f"got exit {result.exit_code}: {result.output}"
    # Output should report it as `image: core` (the canonical group name);
    # the alias is only an input convenience.
    assert "image: core" in result.output
    assert "image: slim" not in result.output  # alias is input-only; output uses canonical name


def test_doctor_expect_unknown_tag_still_errors() -> None:
    """Truly unknown values still get the existing error path."""
    with patch.object(env_resolver, "ENVS_DIR", Path("/nonexistent")):
        result = CliRunner().invoke(app, ["doctor", "--expect", "not-an-image-tag"])
    assert result.exit_code == 2, result.output
    assert "unknown image group" in result.output.lower()


def test_doctor_multi_env_json_mode(tmp_path: Path) -> None:
    """Cross-env aggregation with --json emits a structured envs: [...] payload."""
    envs_dir = _make_fake_envs(tmp_path, ["core", "asr", "tts"])
    # Use a real subprocess so we can split stdout from the stderr-routed table.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    # Inject our ENVS_DIR override by running a small wrapper script.
    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        f"""
from unittest.mock import patch
from pathlib import Path
from voxkitchen.runtime import env_resolver
patch.object(env_resolver, "ENVS_DIR", Path({str(envs_dir)!r})).start()
from voxkitchen.cli.main import app
app(args=["doctor", "--json"], standalone_mode=True)
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(wrapper)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert "envs" in parsed
    assert {e["image_kind"] for e in parsed["envs"]} == {"core", "asr", "tts"}


def test_normalize_text_is_in_core_image_group() -> None:
    """normalize_text is pure-Python (no extras, no model) and should be
    available in the smallest image (slim). Pinning this here so a future
    refactor that drops it from core doesn't silently regress sweep routing."""
    from voxkitchen.cli.doctor import EXPECTED_OPERATORS

    assert "normalize_text" in EXPECTED_OPERATORS["core"], (
        "normalize_text must be in EXPECTED_OPERATORS['core'] so image_for_op "
        "routes it to slim, not latest. See sweep design doc."
    )
