"""Tests for the vkit init scaffolder."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from voxkitchen.cli.main import app


def test_init_creates_project(tmp_path: Path) -> None:
    target = tmp_path / "my-proj"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert (target / "pipeline.yaml").exists()
    assert (target / "README.md").exists()


def test_init_pipeline_yaml_is_valid_yaml(tmp_path: Path) -> None:
    target = tmp_path / "proj"
    runner = CliRunner()
    runner.invoke(app, ["init", str(target)])
    data = yaml.safe_load((target / "pipeline.yaml").read_text())
    assert data["version"] == "0.1"
    assert data["name"] == "my-pipeline"


def test_init_rejects_non_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("x")
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 1
