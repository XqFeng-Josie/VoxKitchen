# VoxKitchen Plan 8: Plugin System + Polish (Final Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the v0.1 feature set: entry_points plugin discovery for third-party operators and recipes, `vkit init` scaffolder, `vkit ingest` standalone command, example pipeline YAMLs, and a basic documentation site. After this plan, VoxKitchen v0.1 is feature-complete.

**Architecture:** The plugin system is a thin lazy-loading layer on top of the existing registries. `vkit init` is a simple directory scaffolder. `vkit ingest` is a standalone wrapper around the existing ingest machinery. No new abstractions — just wiring up the remaining user-facing features.

**Spec reference:** Design spec sections 2 (skeleton), 6 (CLI), 8 (plugins), 9.6 (CI/release), 9.7 (docs).

---

## File Structure

```
src/voxkitchen/
├── plugins/
│   ├── __init__.py
│   └── discovery.py               # NEW: entry_points lazy loading
│
├── cli/
│   ├── main.py                    # MODIFIED: wire init + ingest
│   ├── init_cmd.py                # NEW: vkit init scaffolder
│   └── ingest_cmd.py              # NEW: vkit ingest standalone

examples/
├── pipelines/
│   ├── minimal.yaml               # NEW: identity-only pipeline
│   ├── librispeech-asr.yaml       # NEW: from spec section 5.1
│   └── dir-resample-pack.yaml     # NEW: basic audio processing
└── README.md

docs/
├── index.md                       # NEW: landing page
├── getting-started.md             # NEW: quickstart guide
├── concepts/
│   └── data-protocol.md           # NEW: Recording/Supervision/Cut explained
├── mkdocs.yml                     # NEW: mkdocs config

.github/workflows/
├── ci.yml                         # (unchanged)
├── docs.yml                       # NEW: build + deploy mkdocs to gh-pages
└── release.yml                    # NEW: tag → PyPI + GitHub release

tests/unit/
├── plugins/
│   ├── __init__.py
│   └── test_discovery.py          # NEW
├── cli/
│   ├── test_init_cmd.py           # NEW
│   └── test_ingest_cmd.py         # NEW
```

---

## Task 1: Plugin discovery via entry_points (TDD)

**Files:**
- Create: `src/voxkitchen/plugins/__init__.py`
- Create: `src/voxkitchen/plugins/discovery.py`
- Create: `tests/unit/plugins/__init__.py`
- Create: `tests/unit/plugins/test_discovery.py`
- Modify: `src/voxkitchen/operators/registry.py` — integrate lazy plugin loading

### Implementation

```python
# plugins/discovery.py
"""Lazy entry_points discovery for third-party operators and recipes."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_loaded = False


def load_plugins() -> None:
    """Discover and register third-party operators and recipes via entry_points.

    Called lazily on first access to the operator registry (get_operator).
    Safe to call multiple times — only runs once.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    from importlib.metadata import entry_points

    from voxkitchen.operators.registry import register_operator

    for ep in entry_points(group="voxkitchen.operators"):
        try:
            op_cls = ep.load()
            register_operator(op_cls)
            logger.debug("loaded plugin operator: %s from %s", ep.name, ep.value)
        except Exception:
            logger.warning("failed to load operator plugin: %s", ep.name, exc_info=True)

    from voxkitchen.ingest.recipes import register_recipe

    for ep in entry_points(group="voxkitchen.recipes"):
        try:
            recipe = ep.load()
            register_recipe(recipe)
            logger.debug("loaded plugin recipe: %s from %s", ep.name, ep.value)
        except Exception:
            logger.warning("failed to load recipe plugin: %s", ep.name, exc_info=True)
```

### Integrate into registry

In `operators/registry.py`, modify `get_operator`:
```python
def get_operator(name: str) -> type[Operator]:
    from voxkitchen.plugins.discovery import load_plugins
    load_plugins()  # lazy, runs once
    if name in _REGISTRY:
        return _REGISTRY[name]
    ...
```

### Tests (4)
1. `test_load_plugins_is_idempotent` — call twice, no error
2. `test_load_plugins_discovers_nothing_in_clean_env` — no third-party plugins → built-ins still work
3. `test_get_operator_triggers_plugin_loading` — verify `load_plugins` is called when `get_operator` is used
4. `test_failed_plugin_does_not_crash` — mock a broken entry_point → log warning, other operators still work

### Commit: `feat(plugins): add entry_points lazy discovery for operators and recipes`

---

## Task 2: `vkit init` scaffolder (TDD)

**Files:**
- Create: `src/voxkitchen/cli/init_cmd.py`
- Create: `tests/unit/cli/test_init_cmd.py`
- Modify: `src/voxkitchen/cli/main.py`

### What `vkit init <dir>` does

Creates a minimal pipeline project:
```
<dir>/
├── pipeline.yaml      # template pipeline
└── README.md          # one-liner
```

No interactive prompts in v0.1 — just write a template. Keep it simple.

### Implementation

```python
# cli/init_cmd.py
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
    """Create a minimal pipeline project directory."""
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"directory is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "pipeline.yaml").write_text(PIPELINE_TEMPLATE, encoding="utf-8")
    (target / "README.md").write_text(
        README_TEMPLATE.format(name=target.name), encoding="utf-8"
    )
```

### Wire in main.py

Replace the `init` placeholder:
```python
@app.command(help="Scaffold a new pipeline project directory.")
def init(path: Path = typer.Argument(..., help="Target directory.")) -> None:
    from voxkitchen.cli.init_cmd import init_project
    init_project(path)
    rprint(f"[green]created[/green] pipeline project at {path}")
```

### Tests (3)
1. `test_init_creates_project(tmp_path)` — verify `pipeline.yaml` and `README.md` exist
2. `test_init_pipeline_yaml_is_valid(tmp_path)` — load with `load_pipeline_spec`, should not raise
3. `test_init_rejects_non_empty_dir(tmp_path)` — create a file, run init → FileExistsError

### Commit: `feat(cli): add vkit init scaffolder`

---

## Task 3: `vkit ingest` standalone command (TDD)

**Files:**
- Create: `src/voxkitchen/cli/ingest_cmd.py`
- Create: `tests/unit/cli/test_ingest_cmd.py`
- Modify: `src/voxkitchen/cli/main.py`

### What `vkit ingest` does

Runs an ingest source independently (not as part of a pipeline). Writes the resulting CutSet to a manifest.

```bash
vkit ingest --source dir --out cuts.jsonl.gz --root /data/audio --recursive
vkit ingest --source manifest --out cuts.jsonl.gz --path /existing/manifest.jsonl.gz
vkit ingest --source recipe --out cuts.jsonl.gz --recipe librispeech --root /data/ls
```

### Implementation

```python
# cli/ingest_cmd.py
"""vkit ingest: build a CutSet from a data source."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.ingest import get_ingest_source
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.utils.run_id import generate_run_id


def ingest_command(
    source: str,
    out: Path,
    root: str | None = None,
    path: str | None = None,
    recipe: str | None = None,
    recursive: bool = True,
    subsets: str | None = None,  # comma-separated
) -> None:
    """Run an ingest source and write the result to a manifest."""
    # Build args dict based on source type
    args: dict = {}
    if source == "dir":
        if not root:
            rprint("[red]--root required for source=dir[/red]")
            raise typer.Exit(code=1)
        args = {"root": root, "recursive": recursive}
    elif source == "manifest":
        if not path:
            rprint("[red]--path required for source=manifest[/red]")
            raise typer.Exit(code=1)
        args = {"path": path}
    elif source == "recipe":
        if not recipe or not root:
            rprint("[red]--recipe and --root required for source=recipe[/red]")
            raise typer.Exit(code=1)
        args = {"recipe": recipe, "root": root}
        if subsets:
            args["subsets"] = [s.strip() for s in subsets.split(",")]

    source_cls = get_ingest_source(source)
    config = source_cls.config_cls.model_validate(args)
    run_id = generate_run_id()

    ctx = RunContext(
        work_dir=out.parent, pipeline_run_id=run_id,
        stage_index=0, stage_name="ingest",
        num_gpus=0, num_cpu_workers=1,
        gc_mode="aggressive", device="cpu",
    )

    source_obj = source_cls(config, ctx)
    cuts = source_obj.run()

    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime.now(tz=timezone.utc),
        pipeline_run_id=run_id,
        stage_name="ingest",
    )
    cuts.to_jsonl_gz(out, header)
    rprint(f"[green]wrote {len(cuts)} cuts to {out}[/green]")
```

### Wire in main.py

Replace the `ingest` placeholder:
```python
@app.command(help="Build an initial CutSet from a data source.")
def ingest(
    source: str = typer.Option(..., "--source", help="dir | manifest | recipe"),
    out: Path = typer.Option(..., "--out", help="Output cuts.jsonl.gz path"),
    root: str | None = typer.Option(None, "--root", help="Root directory (for dir/recipe)"),
    path: str | None = typer.Option(None, "--path", help="Manifest path (for source=manifest)"),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe name (for source=recipe)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    subsets: str | None = typer.Option(None, "--subsets", help="Comma-separated subset names"),
) -> None:
    from voxkitchen.cli.ingest_cmd import ingest_command
    ingest_command(source=source, out=out, root=root, path=path, recipe=recipe, recursive=recursive, subsets=subsets)
```

### Tests (3)
1. `test_ingest_dir_writes_manifest(audio_dir, tmp_path)` — `["ingest", "--source", "dir", "--root", audio_dir, "--out", out_path]` → verify manifest exists with 3 cuts
2. `test_ingest_recipe_writes_manifest(mock_librispeech, tmp_path)` — using librispeech recipe
3. `test_ingest_missing_root_exits_1(tmp_path)` — `--source dir` without `--root` → exit 1

### Commit: `feat(cli): add vkit ingest standalone command`

---

## Task 4: Example pipeline YAMLs

**Files:**
- Create: `examples/pipelines/minimal.yaml`
- Create: `examples/pipelines/librispeech-asr.yaml`
- Create: `examples/pipelines/dir-resample-pack.yaml`
- Create: `examples/README.md`

### `minimal.yaml`
```yaml
version: "0.1"
name: minimal
work_dir: ./work/${name}-${run_id}

ingest:
  source: manifest
  args:
    path: ./input.jsonl.gz

stages:
  - name: pass
    op: identity
```

### `librispeech-asr.yaml`
```yaml
version: "0.1"
name: librispeech-asr
description: "Segment LibriSpeech, run ASR, pack to HuggingFace format"
work_dir: ./work/${name}-${run_id}
num_gpus: 4
num_cpu_workers: 16

ingest:
  source: recipe
  recipe: librispeech
  args:
    root: /data/librispeech
    subsets: [train-clean-100, dev-clean]

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }
  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }
  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, language: en, compute_type: float16 }
  - name: snr
    op: snr_estimate
  - name: filter
    op: quality_score_filter
    args: { conditions: ["metrics.snr > 10", "duration > 0.5"] }
  - name: pack
    op: pack_huggingface
```

### `dir-resample-pack.yaml`
```yaml
version: "0.1"
name: dir-resample-pack
description: "Resample local audio to 16kHz mono and pack as Kaldi format"
work_dir: ./work/${name}-${run_id}

ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }
  - name: normalize
    op: loudness_normalize
    args: { target_lufs: -23.0 }
  - name: pack
    op: pack_kaldi
```

### Commit: `docs: add example pipeline YAMLs`

---

## Task 5: Documentation site skeleton (mkdocs)

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/getting-started.md`
- Create: `docs/concepts/data-protocol.md`
- Create: `.github/workflows/docs.yml`

### `mkdocs.yml`
```yaml
site_name: VoxKitchen
site_description: A researcher-friendly speech data processing toolkit
repo_url: https://github.com/voxkitchen/voxkitchen

theme:
  name: material
  palette:
    primary: blue

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Concepts:
    - Data Protocol: concepts/data-protocol.md

markdown_extensions:
  - admonition
  - pymdownx.highlight
  - pymdownx.superfences
```

### `docs/index.md`
```markdown
# VoxKitchen

A researcher-friendly, declarative speech data processing toolkit.

## Features

- **Declarative pipelines**: Write YAML, run with `vkit run`
- **22 built-in operators**: Segmentation, ASR, diarization, quality filtering, packaging
- **Resumable**: Checkpoint every stage, resume after crashes
- **Disk-aware**: Aggressive GC for derived audio files
- **Inspectable**: Rich CLI + HTML report + Gradio panel
- **Extensible**: Plugin system for third-party operators and recipes

## Quick start

```bash
pip install voxkitchen
vkit init my-project
cd my-project
vkit run pipeline.yaml
```
```

### `docs/getting-started.md`
Quick install, first pipeline, inspect results. Keep it under 100 lines.

### `docs/concepts/data-protocol.md`
Explain Recording, Supervision, Cut, CutSet, Provenance, JSONL.gz format. Keep it under 150 lines.

### `.github/workflows/docs.yml`
```yaml
name: docs
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install mkdocs-material
      - run: mkdocs gh-deploy --force
```

### Commit: `docs: add mkdocs documentation site skeleton`

---

## Task 6: Release workflow

**Files:**
- Create: `.github/workflows/release.yml`

### `release.yml`
```yaml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # for PyPI trusted publishing
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Also create a GitHub Release from the tag (can be manual or via `gh release create` in CI).

### Commit: `ci: add release workflow for PyPI publishing`

---

## Task 7: Final cleanup + smoke test all CLI commands

**Files:** Various small fixes

- [ ] Remove the `_not_implemented` helper from `main.py` if no commands use it
- [ ] Update `test_placeholder_commands_exit_with_code_1` — `init` and `ingest` are now real commands, update or remove the test
- [ ] Run `vkit init /tmp/test-project && vkit validate /tmp/test-project/pipeline.yaml` — verify the scaffolded pipeline is valid
- [ ] Run `vkit ingest --source dir --root <audio_dir> --out /tmp/test.jsonl.gz` — verify it works
- [ ] Run `vkit inspect cuts /tmp/test.jsonl.gz` — verify output

### Commit: `chore: final cleanup — remove placeholders, update smoke tests`

---

## Task 8: Full verification + tag

- [ ] `pytest -m "not slow and not gpu"` — all tests pass
- [ ] `ruff check src tests` + `ruff format --check`
- [ ] `mypy src/voxkitchen tests`
- [ ] `pre-commit run --all-files`
- [ ] `vkit --help` — all 6 commands are real (no "not implemented")
- [ ] `vkit init`, `vkit ingest`, `vkit validate`, `vkit run`, `vkit inspect`, `vkit viz` — all respond correctly
- [ ] Example YAMLs pass `vkit validate`
- [ ] Tag: `git tag -a v0.1.0-rc1 -m "VoxKitchen v0.1.0 release candidate 1"`

---

## Plan 8 Completion Checklist

- [ ] Plugin discovery loads third-party operators/recipes via `entry_points`
- [ ] Plugin errors are logged but don't crash the application
- [ ] `vkit init <dir>` creates a valid pipeline project
- [ ] `vkit ingest --source dir/manifest/recipe` works standalone
- [ ] Example YAMLs in `examples/pipelines/` are valid
- [ ] mkdocs site builds (`mkdocs build` succeeds)
- [ ] `.github/workflows/docs.yml` deploys to gh-pages on push to main
- [ ] `.github/workflows/release.yml` publishes to PyPI on `v*` tags
- [ ] All CLI commands are real (no placeholders remain)
- [ ] All existing tests pass
- [ ] `git tag v0.1.0-rc1`

---

## v0.1 Feature Completeness After Plan 8

| Feature | Status |
|---|---|
| Data protocol (Recording/Supervision/Cut/CutSet) | Done (Plan 1) |
| Pipeline engine (YAML, runner, executors, GC) | Done (Plan 2) |
| DirScan + ManifestImport ingest | Done (Plans 2-3) |
| 22 operators (basic/segment/annotate/quality/pack) | Done (Plans 3-5) |
| 3 ingest recipes (LibriSpeech/CommonVoice/AISHELL) | Done (Plan 6) |
| Rich CLI inspect | Done (Plan 7) |
| HTML report | Done (Plan 7) |
| Gradio panel | Done (Plan 7) |
| Plugin system (entry_points) | Plan 8 |
| `vkit init` scaffolder | Plan 8 |
| `vkit ingest` standalone | Plan 8 |
| Example pipelines | Plan 8 |
| Documentation site | Plan 8 |
| Release workflow | Plan 8 |

After Plan 8: **v0.1.0-rc1 ready for release.**
