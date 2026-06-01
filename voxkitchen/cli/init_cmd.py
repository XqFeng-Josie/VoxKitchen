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

# Where stage outputs land. ${run_id} disambiguates parallel runs.
work_dir: ./work/${name}-${run_id}

# Where audio comes from. `source: dir` recursively walks ./data for
# audio files. Other options: `recipe` (for catalogued datasets,
# see `vkit datasets --recipe-only`) and `manifest` (an existing CutSet).
ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  # 1. Normalise sample rate + channels — most downstream ops assume 16k mono.
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1

  # 2. Write the final CutSet manifest. Add operators between
  #    `resample` and `pack` to actually do work; discover them with
  #    `vkit operators` (or `vkit operators search <keyword>`).
  - name: pack
    op: pack_manifest
"""
)

README_TEMPLATE = """\
# {name}

A VoxKitchen pipeline project. Edit `pipeline.yaml` to declare the
processing chain you want, then iterate with the commands below.

## First run

```bash
# 1. Put audio files under ./data/
cp /path/to/audio/* data/

# 2. (Recommended) preview what your pipeline does — chain + each stage's
#    reads/writes contract — before kicking off a run.
vkit show pipeline.yaml

# 3. Validate the YAML and arg schemas (catches typos like
#    `target_channel: 1` → suggests `target_channels`).
vkit validate pipeline.yaml

# 4. Execute. The dry-run validates inside the Docker image; the real
#    run does the work and prints the exact work_dir afterwards.
vkit docker run --tag {tag} pipeline.yaml --dry-run
vkit docker run --tag {tag} pipeline.yaml
```

## After a run

```bash
# Stage-by-stage status (which stages completed, how big each manifest is).
vkit inspect run <work_dir>

# Statistics for the final CutSet (durations, languages, gender, metrics).
vkit inspect cuts <work_dir>/<final_stage>/cuts.jsonl.gz

# A standalone HTML showcase card you can share / commit to a repo.
vkit card <work_dir>/<final_stage>/cuts.jsonl.gz --out card.html
```

## Iteration tips

- `vkit operators` (and `vkit operators search <keyword>`) lists the 50+
  built-in operators with their reads/writes contracts so you can pick
  the right one for the gap in your chain.
- `vkit datasets` lets you browse the catalog of public datasets you can
  drop into `ingest: {{ source: recipe, recipe: <name> }}` to skip the
  manual download step.
- Re-running with `--resume-from <stage>` skips already-completed stages
  — useful when you're tuning a single operator's args.
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

    # Create data directory. Drop a .gitkeep so the directory survives
    # ``git add .`` — otherwise users committing the scaffolded project
    # find that the README's ``Put audio under ./data/`` instruction
    # refers to a directory their teammates can't see.
    data_dir = target / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / ".gitkeep").touch()

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
