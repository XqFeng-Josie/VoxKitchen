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


def test_init_template_writes_packaged_pipeline(tmp_path: Path) -> None:
    target = tmp_path / "cleaning-proj"
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(target), "--template", "cleaning"])
    assert result.exit_code == 0

    data = yaml.safe_load((target / "pipeline.yaml").read_text())
    assert data["name"] == "data-cleaning"
    assert data["stages"][-1]["op"] == "pack_jsonl"
    readme = (target / "README.md").read_text()
    assert "vkit docker run --tag slim pipeline.yaml --dry-run" in readme
    assert "vkit run pipeline.yaml" not in readme


def test_init_scaffolds_data_dir_with_gitkeep(tmp_path: Path) -> None:
    """data/ is created with a .gitkeep so it survives `git add` for users
    who commit the scaffolded project."""
    target = tmp_path / "p"
    CliRunner().invoke(app, ["init", str(target)])
    assert (target / "data").is_dir()
    assert (target / "data" / ".gitkeep").is_file()


def test_init_default_pipeline_has_inline_comments(tmp_path: Path) -> None:
    """The default pipeline must include comments explaining what each
    stage does and pointing at the discovery commands. Without them the
    first-time user is just reading raw YAML."""
    target = tmp_path / "p"
    CliRunner().invoke(app, ["init", str(target)])
    text = (target / "pipeline.yaml").read_text()
    assert "vkit operators" in text  # discovery hint
    assert "16k mono" in text  # the why for the resample step


def test_init_readme_points_at_show_validate_card(tmp_path: Path) -> None:
    """README guides the iteration loop: show → validate → run → inspect → card."""
    target = tmp_path / "p"
    CliRunner().invoke(app, ["init", str(target)])
    readme = (target / "README.md").read_text()
    for needle in [
        "vkit show pipeline.yaml",
        "vkit validate pipeline.yaml",
        "vkit inspect cuts",
        "vkit card",
        "vkit operators",
        "vkit datasets",
    ]:
        assert needle in readme, f"README missing iteration hint: {needle}"


def test_init_next_steps_output_lists_show_and_validate(tmp_path: Path) -> None:
    """The CLI 'Next steps' echo now includes the show + validate commands
    so users see the iteration loop without having to open the README."""
    target = tmp_path / "p"
    result = CliRunner().invoke(app, ["init", str(target)])
    assert result.exit_code == 0
    assert "vkit show pipeline.yaml" in result.output
    assert "vkit validate pipeline.yaml" in result.output
    assert "vkit operators search" in result.output


def test_init_rejects_non_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("x")
    runner = CliRunner()
    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 1


def test_init_default_pipeline_starts_with_schema_directive(tmp_path: Path) -> None:
    target = tmp_path / "p"
    CliRunner().invoke(app, ["init", str(target)])
    first_line = (target / "pipeline.yaml").read_text().splitlines()[0]
    assert first_line.startswith("# yaml-language-server: $schema=")
    assert "docs/schemas/pipeline.schema.json" in first_line


def test_init_template_pipeline_starts_with_schema_directive(tmp_path: Path) -> None:
    # Templates ship without the schema header (so editing them in the repo
    # stays clean); init_project injects it at scaffold time. Verify it's
    # present in the scaffolded copy.
    target = tmp_path / "p"
    CliRunner().invoke(app, ["init", str(target), "--template", "asr"])
    first_line = (target / "pipeline.yaml").read_text().splitlines()[0]
    assert first_line.startswith("# yaml-language-server: $schema=")
