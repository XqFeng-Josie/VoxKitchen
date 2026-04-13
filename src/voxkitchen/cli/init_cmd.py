"""vkit init: scaffold a new pipeline project."""

from __future__ import annotations

from pathlib import Path

PIPELINE_TEMPLATE = """\
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


def init_project(target: Path) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "pipeline.yaml").write_text(PIPELINE_TEMPLATE, encoding="utf-8")
    (target / "README.md").write_text(README_TEMPLATE.format(name=target.name), encoding="utf-8")
