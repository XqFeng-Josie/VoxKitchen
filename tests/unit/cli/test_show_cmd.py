"""Tests for ``vkit show <pipeline.yaml>``.

The command is presentation-only — it loads a YAML, looks each operator
contract up via the same code path ``vkit validate`` uses, and prints the
result. These tests pin the visible contract (header fields, stage layout,
contract rows, the two distinct "no contract" footnotes) so accidental
changes in the rendering get caught.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.cli.show_cmd import _contract_for


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Helper coverage — separates registry / schemas / unknown branches so the
# CLI tests can stay focused on visible output.
# ---------------------------------------------------------------------------


def test_contract_for_registered_op_returns_found_true() -> None:
    """A built-in operator's class-attr contract surfaces with found=True."""
    contract, found = _contract_for("resample", None)
    assert found is True
    assert "audio" in contract["reads"]
    assert "custom.origin_start" in contract["writes"]


def test_contract_for_registered_filter_op_is_empty_but_found() -> None:
    """Filter operators (e.g. quality_score_filter) declare empty contracts
    by design — found must still be True so the show command can tell them
    apart from genuinely-unknown operators."""
    contract, found = _contract_for("quality_score_filter", None)
    assert found is True
    assert contract == {"reads": [], "writes": [], "optional_reads": [], "clears": []}


def test_contract_for_unknown_op_returns_found_false() -> None:
    contract, found = _contract_for("does_not_exist_xyz", None)
    assert found is False
    assert contract == {"reads": [], "writes": [], "optional_reads": [], "clears": []}


def test_contract_for_schemas_fallback() -> None:
    """When the operator is not in the registry, fall back to op_schemas.json."""
    schemas = {
        "ghost_op": {
            "reads": ["audio"],
            "writes": ["metrics.ghost"],
            "device": "cpu",
        }
    }
    contract, found = _contract_for("ghost_op", schemas)
    assert found is True
    assert contract["reads"] == ["audio"]
    assert contract["writes"] == ["metrics.ghost"]


# ---------------------------------------------------------------------------
# End-to-end CLI surface — pins the contract block, args row, and footnotes.
# ---------------------------------------------------------------------------


def test_show_renders_header_and_stage_count(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: smoke
description: pretty-print smoke test
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp/data } }
stages:
  - { name: r, op: resample, args: { target_sr: 16000, target_channels: 1 } }
  - { name: q, op: quality_score_filter, args: { conditions: [] } }
""",
    )
    result = CliRunner().invoke(app, ["show", str(yaml_path)])
    assert result.exit_code == 0, result.output
    # Header (Rich panel)
    assert "smoke" in result.output
    assert "pretty-print smoke test" in result.output
    assert "ingest" in result.output and "dir" in result.output
    # Stage count + numbering
    assert "Stages (2)" in result.output
    assert "01" in result.output and "02" in result.output


def test_show_prints_reads_and_writes_for_registered_op(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: c
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - { name: r, op: resample, args: { target_sr: 16000, target_channels: 1 } }
""",
    )
    result = CliRunner().invoke(app, ["show", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert "reads:" in result.output
    assert "audio" in result.output
    assert "writes:" in result.output
    assert "custom.origin_start" in result.output


def test_show_marks_unknown_operator_softly(tmp_path: Path) -> None:
    """An unknown operator must surface a 'not available in this env' note —
    distinct from the 'no static contract' note for known-but-empty filters."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: u
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - { name: ghost, op: this_op_does_not_exist }
""",
    )
    result = CliRunner().invoke(app, ["show", str(yaml_path)])
    # Show is presentation-only — it does NOT fail on unknown ops.
    assert result.exit_code == 0, result.output
    assert "not available in this env" in result.output
    assert "this_op_does_not_exist" in result.output


def test_show_filter_op_uses_no_static_contract_note(tmp_path: Path) -> None:
    """quality_score_filter is registered but declares empty contracts —
    must surface the 'no static contract' note, NOT 'not available'."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: f
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - { name: q, op: quality_score_filter, args: { conditions: [] } }
""",
    )
    result = CliRunner().invoke(app, ["show", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert "no static contract" in result.output
    # NOT the unknown-op message.
    assert "not available in this env" not in result.output


def test_show_invalid_yaml_exits_1(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("not: [valid yaml :::: \n", encoding="utf-8")
    result = CliRunner().invoke(app, ["show", str(bad)])
    assert result.exit_code == 1
    assert "error" in result.output.lower()


def test_show_next_steps_hint_appears(tmp_path: Path) -> None:
    """Show closes with a hint pointing to validate / run for discoverability."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: n
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - { name: r, op: resample, args: { target_sr: 16000, target_channels: 1 } }
""",
    )
    result = CliRunner().invoke(app, ["show", str(yaml_path)])
    assert result.exit_code == 0
    assert "vkit validate" in result.output
    assert "vkit run" in result.output
