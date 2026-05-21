"""vkit init: scaffold a new pipeline project."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint
from rich.table import Table

# Editors that understand the YAML Language Server protocol (the VS Code
# "YAML" extension, Neovim's yamlls, JetBrains' YAML support) fetch this
# URL once and provide autocompletion + hover docs + inline validation while
# users edit the file. The schema is checked into the repo and served from
# raw.githubusercontent.com so it works without a custom hosting setup.
SCHEMA_HEADER = (
    "# yaml-language-server: $schema="
    "https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen"
    "/main/docs/schemas/pipeline.schema.json\n"
)

DEFAULT_PIPELINE = (
    SCHEMA_HEADER
    + """\
version: "0.1"
name: my-pipeline
description: "A VoxKitchen pipeline"

work_dir: ./work/${name}-${run_id}

ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1

  - name: pack
    op: pack_manifest
"""
)

README_TEMPLATE = """\
# {name}

A VoxKitchen pipeline project.

## Usage

```bash
# Put audio files under data/ first.
vkit docker run --tag {tag} pipeline.yaml --dry-run
vkit docker run --tag {tag} pipeline.yaml

# `vkit docker run` prints the exact work_dir for the run.
vkit inspect run <work_dir>
vkit inspect cuts <work_dir>/<final_stage>/cuts.jsonl.gz
```
"""


def recommended_docker_tag(template: str | None) -> str:
    """Return the recommended prebuilt Docker image tag for a scaffolded project."""
    if template == "asr":
        return "asr"
    if template == "cleaning" or template is None:
        return "slim"
    return "latest"


def list_templates() -> None:
    """Print all available pipeline templates."""
    from voxkitchen.templates import TEMPLATES

    t = Table(title="Available pipeline templates")
    t.add_column("Name", style="bold")
    t.add_column("Description")
    t.add_column("Usage")

    for name, info in sorted(TEMPLATES.items()):
        t.add_row(name, info["description"], f"vkit init <path> --template {name}")

    rprint(t)
    rprint()
    rprint("[dim]Example:[/dim] vkit init my-project --template tts")


def init_project(target: Path, template: str | None = None) -> None:
    """Create a new pipeline project directory.

    Args:
        target: Directory to create.
        template: Optional template name (tts, asr, cleaning, speaker).
            If None, uses a minimal default pipeline.
    """
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)

    # Create data directory
    (target / "data").mkdir(exist_ok=True)

    if template is not None:
        from voxkitchen.templates import get_template_content

        pipeline_content = get_template_content(template)
        # Templates ship without the schema header so editing them in the
        # repo stays clean; injected here so every scaffolded project gets
        # editor autocomplete out of the box.
        if not pipeline_content.lstrip().startswith("# yaml-language-server:"):
            pipeline_content = SCHEMA_HEADER + pipeline_content
    else:
        pipeline_content = DEFAULT_PIPELINE

    (target / "pipeline.yaml").write_text(pipeline_content, encoding="utf-8")
    tag = recommended_docker_tag(template)
    (target / "README.md").write_text(
        README_TEMPLATE.format(name=target.name, tag=tag), encoding="utf-8"
    )
