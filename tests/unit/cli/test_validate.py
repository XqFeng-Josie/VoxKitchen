"""Unit tests for the real `vkit validate` command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from voxkitchen.cli.main import app


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_validate_accepts_valid_pipeline(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: s0, op: identity }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_rejects_unknown_operator(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: s0, op: not_a_real_operator }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1
    assert "not_a_real_operator" in result.output


def test_validate_rejects_malformed_yaml(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path, "version: 0.1\nstages: [bad: syntax")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1
