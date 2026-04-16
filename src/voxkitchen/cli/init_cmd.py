"""vkit init: scaffold a new pipeline project."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint
from rich.table import Table

DEFAULT_PIPELINE = """\
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

README_TEMPLATE = """\
# {name}

A VoxKitchen pipeline project.

## Usage

```bash
vkit validate pipeline.yaml
vkit run pipeline.yaml
vkit inspect run work/
```
"""


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
    else:
        pipeline_content = DEFAULT_PIPELINE

    (target / "pipeline.yaml").write_text(pipeline_content, encoding="utf-8")
    (target / "README.md").write_text(README_TEMPLATE.format(name=target.name), encoding="utf-8")
