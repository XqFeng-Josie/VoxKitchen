# VoxKitchen Plan 1: Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the VoxKitchen Python project and implement the complete data protocol (Recording, Supervision, Cut, CutSet, Provenance, JSONL.gz I/O) with full unit test coverage, plus a placeholder CLI that runs.

**Architecture:** Standard `src/` Python package layout with Pydantic v2 models for the data schema. CutSet is a thin lazy wrapper over JSONL.gz streams. TDD throughout — every model and I/O function gets its tests written first. The CLI is a Typer app skeleton with 6 placeholder subcommands that print "not yet implemented" — real behavior is filled in by later plans.

**Tech Stack:** Python 3.10+, Pydantic v2, Typer, pytest, ruff, mypy, hatchling (build backend), soundfile (audio I/O), PyYAML.

**Spec reference:** `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/docs/superpowers/specs/2026-04-11-voxkitchen-design.md` — sections 1, 2, 3, 6, 9.

---

## File Structure Produced by This Plan

```
SpeechDatasetHub/                           # repo root (already exists, empty)
├── pyproject.toml                          # packaging + deps + tools config
├── LICENSE                                 # Apache 2.0
├── README.md                               # minimal project intro
├── .gitignore
├── .pre-commit-config.yaml
├── src/voxkitchen/
│   ├── __init__.py                         # __version__
│   ├── __main__.py                         # `python -m voxkitchen` entry
│   ├── schema/
│   │   ├── __init__.py                     # re-exports public types
│   │   ├── provenance.py                   # Provenance model
│   │   ├── recording.py                    # AudioSource + Recording models
│   │   ├── supervision.py                  # Supervision model
│   │   ├── cut.py                          # Cut model
│   │   ├── io.py                           # header record + jsonl.gz read/write
│   │   └── cutset.py                       # CutSet wrapper with lazy I/O
│   ├── utils/
│   │   ├── __init__.py
│   │   └── time.py                         # now_utc() helper
│   └── cli/
│       ├── __init__.py
│       └── main.py                         # Typer app with 6 placeholder commands
├── tests/
│   ├── __init__.py
│   ├── conftest.py                         # pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_cli_smoke.py
│   │   └── schema/
│   │       ├── __init__.py
│   │       ├── test_provenance.py
│   │       ├── test_recording.py
│   │       ├── test_supervision.py
│   │       ├── test_cut.py
│   │       ├── test_io.py
│   │       └── test_cutset.py
│   └── fixtures/
│       └── .gitkeep
└── .github/
    └── workflows/
        └── ci.yml                          # ruff + mypy + pytest on 3.10/3.11/3.12
```

**Responsibility per file:**
- `schema/provenance.py` — Provenance dataclass: where did a Cut come from?
- `schema/recording.py` — AudioSource + Recording: physical audio, never mutated
- `schema/supervision.py` — Supervision: labels over a time interval of a Recording
- `schema/cut.py` — Cut: trainable sample referencing a Recording slice with Supervisions
- `schema/io.py` — header record, serialization helpers, `CutReader`/`CutWriter` for JSONL.gz with header
- `schema/cutset.py` — `CutSet` lazy wrapper: in-memory list or streaming from disk; `split` / `filter` / `map` / `merge` / `concat_from_disk`
- `utils/time.py` — `now_utc()` for deterministic UTC timestamps
- `cli/main.py` — Typer app with 6 placeholder subcommands that print "not yet implemented"
- Tests: one test module per schema module; `test_cli_smoke.py` verifies `vkit --help` runs

---

## Task 1: Bootstrap repo, git, and project metadata

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `pyproject.toml`

- [ ] **Step 1: Initialize git repository**

The repo root is `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/`. It already contains `docs/superpowers/specs/` and `docs/superpowers/plans/` from the brainstorming phase but is not a git repo.

Run (from the repo root):
```bash
cd /Users/mobvoi/Downloads/USC/1/SpeechDatasetHub
git init
git branch -M main
```

Expected: `Initialized empty Git repository in ...`

- [ ] **Step 2: Write `.gitignore`**

Create `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
*.egg
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual environments
.venv/
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
.DS_Store

# Project-specific
work_dir/
*.jsonl.gz
!tests/fixtures/**/*.jsonl.gz
```

- [ ] **Step 3: Write `LICENSE` (Apache 2.0)**

Create `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/LICENSE` with the standard Apache 2.0 text. Fetch it verbatim from the official source, or write this header plus include the full license body:

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

Copyright 2026 VoxKitchen contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

**Note:** the full Apache 2.0 license text (with the "TERMS AND CONDITIONS" body) is required for compliance. Download from https://www.apache.org/licenses/LICENSE-2.0.txt and replace this file's contents with the full text, keeping the `Copyright 2026 VoxKitchen contributors` line.

- [ ] **Step 4: Write minimal `README.md`**

Create `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/README.md`:

```markdown
# VoxKitchen

A researcher-friendly, declarative speech data processing toolkit with a unified data protocol.

> **Status:** Pre-alpha. This repository is under active development and the API is unstable.

## What it does

VoxKitchen takes raw audio (local directories, manifests, or open-source datasets) and runs it through declarative YAML pipelines that segment, auto-label, quality-filter, and package it for training. Every intermediate step is inspectable, every run is resumable, and every output carries full provenance.

## Installation

```bash
pip install voxkitchen
```

## Quickstart

```bash
vkit --help
```

Full documentation coming soon. See `docs/superpowers/specs/` for the design spec.

## License

Apache 2.0. See [LICENSE](LICENSE).
```

- [ ] **Step 5: Write `pyproject.toml`**

Create `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/pyproject.toml`:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "voxkitchen"
dynamic = ["version"]
description = "A researcher-friendly, declarative speech data processing toolkit"
readme = "README.md"
requires-python = ">=3.10"
license = { file = "LICENSE" }
authors = [{ name = "VoxKitchen contributors" }]
keywords = ["speech", "audio", "dataset", "pipeline", "asr"]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Topic :: Multimedia :: Sound/Audio",
]

# Plan 1 only uses a subset of core deps (schema + CLI).
# Later plans will add torch, torchaudio, silero-vad, etc. per spec section 9.2.
dependencies = [
    "pydantic>=2.5,<3",
    "pyyaml>=6",
    "typer>=0.12",
    "rich>=13",
    "soundfile>=0.12",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pre-commit>=3.6",
]

[project.scripts]
vkit = "voxkitchen.cli.main:app"

[project.urls]
Repository = "https://github.com/voxkitchen/voxkitchen"  # placeholder

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/voxkitchen"]

[tool.hatch.build.hooks.vcs]
version-file = "src/voxkitchen/_version.py"

[tool.ruff]
line-length = 100
target-version = "py310"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "UP",  # pyupgrade
    "RUF", # ruff-specific
]
ignore = [
    "E501",  # line-length handled by formatter
]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
files = ["src/voxkitchen", "tests"]

[[tool.mypy.overrides]]
module = ["soundfile.*", "yaml.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra --strict-markers --strict-config"
testpaths = ["tests"]
pythonpath = ["src"]
```

**Note on the `dependencies` list:** Plan 1 deliberately uses a minimal subset. Spec section 9.2 Plan B includes `torch`, `torchaudio`, `ffmpeg-python`, `silero-vad`, etc. These will be added in Plan 2 or Plan 3 when operators that need them are introduced. Deferring heavy deps keeps Plan 1's install fast.

- [ ] **Step 6: Commit the bootstrap**

```bash
git add .gitignore LICENSE README.md pyproject.toml docs/
git commit -m "chore: bootstrap voxkitchen project with metadata and spec"
```

Expected: one commit with 4 new files plus the previously-written spec document.

---

## Task 2: Create source layout and verify install

**Files:**
- Create: `src/voxkitchen/__init__.py`
- Create: `src/voxkitchen/__main__.py`
- Create: `src/voxkitchen/schema/__init__.py`
- Create: `src/voxkitchen/utils/__init__.py`
- Create: `src/voxkitchen/cli/__init__.py`
- Create: `src/voxkitchen/cli/main.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_cli_smoke.py`
- Create: `tests/unit/schema/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/voxkitchen/schema
mkdir -p src/voxkitchen/utils
mkdir -p src/voxkitchen/cli
mkdir -p tests/unit/schema
mkdir -p tests/fixtures
touch tests/fixtures/.gitkeep
```

- [ ] **Step 2: Write `src/voxkitchen/__init__.py`**

```python
"""VoxKitchen — declarative speech data processing toolkit."""

try:
    from voxkitchen._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
```

- [ ] **Step 3: Write `src/voxkitchen/__main__.py`**

```python
"""Enable `python -m voxkitchen` to invoke the CLI."""

from voxkitchen.cli.main import app

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Write `src/voxkitchen/schema/__init__.py`**

Leave empty for now — each schema module will be re-exported here as it lands.

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""
```

- [ ] **Step 5: Write `src/voxkitchen/utils/__init__.py`**

```python
"""Internal utility modules shared across voxkitchen packages."""
```

- [ ] **Step 6: Write `src/voxkitchen/cli/__init__.py`**

```python
"""voxkitchen command-line interface."""
```

- [ ] **Step 7: Write `src/voxkitchen/cli/main.py` with Typer placeholder app**

```python
"""Top-level Typer application exposing the `vkit` CLI.

Plan 1 ships a placeholder skeleton: each subcommand prints a
"not yet implemented" message. Later plans replace these stubs with
real behavior.
"""

from __future__ import annotations

import typer
from rich import print as rprint

app = typer.Typer(
    name="vkit",
    help="VoxKitchen — declarative speech data processing toolkit.",
    no_args_is_help=True,
    add_completion=False,
)


def _not_implemented(command: str) -> None:
    rprint(f"[yellow]vkit {command}[/yellow]: not yet implemented in this build.")
    raise typer.Exit(code=1)


@app.command(help="Scaffold a new pipeline project directory.")
def init(path: str = typer.Argument(..., help="Target directory.")) -> None:
    _not_implemented(f"init {path}")


@app.command(help="Build an initial CutSet from a data source.")
def ingest(
    source: str = typer.Option(..., "--source", help="dir | manifest | recipe"),
    out: str = typer.Option(..., "--out", help="Output cuts.jsonl.gz path"),
) -> None:
    _not_implemented(f"ingest --source {source} --out {out}")


@app.command(help="Parse and validate a pipeline YAML (no execution).")
def validate(pipeline: str = typer.Argument(..., help="Pipeline YAML path.")) -> None:
    _not_implemented(f"validate {pipeline}")


@app.command(help="Execute a pipeline.")
def run(pipeline: str = typer.Argument(..., help="Pipeline YAML path.")) -> None:
    _not_implemented(f"run {pipeline}")


@app.command(help="Inspect cuts, recordings, run progress, trace, or errors.")
def inspect(
    subcommand: str = typer.Argument(..., help="cuts | recordings | run | trace | errors"),
    path: str = typer.Argument(..., help="Target path."),
) -> None:
    _not_implemented(f"inspect {subcommand} {path}")


@app.command(help="Launch local Gradio panel to explore a CutSet.")
def viz(path: str = typer.Argument(..., help="CutSet or work_dir path.")) -> None:
    _not_implemented(f"viz {path}")


if __name__ == "__main__":
    app()
```

- [ ] **Step 8: Write `tests/__init__.py`, `tests/unit/__init__.py`, `tests/unit/schema/__init__.py`**

All three files are empty `.py` files that mark the directories as Python packages. Create them all empty.

- [ ] **Step 9: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures for voxkitchen tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def fixed_datetime() -> datetime:
    """A deterministic UTC datetime for reproducible tests."""
    return datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def tmp_jsonl_gz(tmp_path: Path) -> Path:
    """Return a path to a temporary .jsonl.gz file inside tmp_path."""
    return tmp_path / "cuts.jsonl.gz"
```

- [ ] **Step 10: Write `tests/unit/test_cli_smoke.py`**

```python
"""Smoke tests: `vkit --help` runs and exits cleanly."""

from __future__ import annotations

from typer.testing import CliRunner

from voxkitchen.cli.main import app


def test_top_level_help_runs() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "vkit" in result.output.lower()


def test_six_top_level_commands_are_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    expected_commands = {"init", "ingest", "validate", "run", "inspect", "viz"}
    for cmd in expected_commands:
        assert cmd in result.output, f"command '{cmd}' missing from --help output"
```

- [ ] **Step 11: Install the project in editable mode**

Create and activate a virtual environment first if one doesn't exist:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Expected: pip installs pydantic, typer, rich, soundfile, pyyaml, tqdm, pytest, ruff, mypy, pre-commit, and voxkitchen itself. Install should complete in under 30 seconds.

- [ ] **Step 12: Verify `vkit --help` runs**

```bash
vkit --help
```

Expected output contains: "VoxKitchen", "init", "ingest", "validate", "run", "inspect", "viz".

- [ ] **Step 13: Run the CLI smoke tests**

```bash
pytest tests/unit/test_cli_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 14: Commit**

```bash
git add src/voxkitchen tests/
git commit -m "feat: add src layout, placeholder CLI, and smoke test"
```

---

## Task 3: Implement Provenance model (TDD)

**Files:**
- Create: `tests/unit/schema/test_provenance.py`
- Create: `src/voxkitchen/utils/time.py`
- Create: `src/voxkitchen/schema/provenance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_provenance.py`:

```python
"""Unit tests for voxkitchen.schema.provenance.Provenance."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voxkitchen.schema.provenance import Provenance


def test_provenance_minimal_construction() -> None:
    p = Provenance(
        source_cut_id="cut-parent",
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    assert p.source_cut_id == "cut-parent"
    assert p.generated_by == "silero_vad@0.4.1"
    assert p.stage_name == "02_vad"
    assert p.pipeline_run_id == "run-a1b2c3"


def test_provenance_source_cut_id_can_be_none_for_root_cuts() -> None:
    p = Provenance(
        source_cut_id=None,
        generated_by="dir_scan",
        stage_name="00_ingest",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    assert p.source_cut_id is None


def test_provenance_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Provenance(  # type: ignore[call-arg]
            generated_by="silero_vad",
            stage_name="02_vad",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        )


def test_provenance_round_trips_through_json() -> None:
    original = Provenance(
        source_cut_id="cut-parent",
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    blob = original.model_dump_json()
    restored = Provenance.model_validate_json(blob)
    assert restored == original


def test_provenance_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Provenance.model_validate(
            {
                "source_cut_id": "cut-parent",
                "generated_by": "silero_vad",
                "stage_name": "02_vad",
                "created_at": "2026-04-11T10:30:00Z",
                "pipeline_run_id": "run-a1b2c3",
                "surprise_field": "boom",
            }
        )
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_provenance.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.provenance'` or similar.

- [ ] **Step 3: Write `src/voxkitchen/utils/time.py`**

```python
"""Deterministic UTC datetime helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current UTC datetime with tzinfo attached.

    Wrapped in a function so tests can monkeypatch it when determinism matters.
    """
    return datetime.now(tz=timezone.utc)
```

- [ ] **Step 4: Write `src/voxkitchen/schema/provenance.py`**

```python
"""Provenance record: describes how a Cut was produced."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Provenance(BaseModel):
    """Where this Cut came from in the pipeline.

    - ``source_cut_id`` is ``None`` only for freshly-ingested Cuts that have no
      parent in the VoxKitchen data model. Every derived Cut (VAD output,
      filtered subset, etc.) points back to its parent.
    - ``generated_by`` is an operator identifier, conventionally
      ``<operator_name>@<version>`` (e.g. ``"silero_vad@0.4.1"``).
    - ``stage_name`` matches the stage name in the pipeline YAML, so
      ``vkit inspect`` can cross-reference with the run's ``run.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    source_cut_id: str | None
    generated_by: str
    stage_name: str
    created_at: datetime
    pipeline_run_id: str
```

- [ ] **Step 5: Re-export from `src/voxkitchen/schema/__init__.py`**

Edit `src/voxkitchen/schema/__init__.py` to add:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.provenance import Provenance

__all__ = ["Provenance"]
```

- [ ] **Step 6: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_provenance.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/schema/provenance.py src/voxkitchen/schema/__init__.py src/voxkitchen/utils/time.py tests/unit/schema/test_provenance.py
git commit -m "feat(schema): add Provenance model with unit tests"
```

---

## Task 4: Implement AudioSource and Recording models (TDD)

**Files:**
- Create: `tests/unit/schema/test_recording.py`
- Create: `src/voxkitchen/schema/recording.py`
- Modify: `src/voxkitchen/schema/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_recording.py`:

```python
"""Unit tests for voxkitchen.schema.recording.Recording and AudioSource."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxkitchen.schema.recording import AudioSource, Recording


def test_audio_source_file_construction() -> None:
    src = AudioSource(type="file", channels=[0], source="/data/foo.wav")
    assert src.type == "file"
    assert src.channels == [0]
    assert src.source == "/data/foo.wav"


def test_audio_source_url_construction() -> None:
    src = AudioSource(
        type="url",
        channels=[0, 1],
        source="https://example.com/audio.flac",
    )
    assert src.type == "url"


def test_audio_source_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        AudioSource(type="carrier_pigeon", channels=[0], source="wat")  # type: ignore[arg-type]


def test_recording_minimal_construction() -> None:
    rec = Recording(
        id="librispeech-1089-134686-0001",
        sources=[AudioSource(type="file", channels=[0], source="/data/foo.wav")],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
        num_channels=1,
    )
    assert rec.id == "librispeech-1089-134686-0001"
    assert rec.duration == 10.0
    assert rec.checksum is None
    assert rec.custom == {}


def test_recording_accepts_checksum_and_custom() -> None:
    rec = Recording(
        id="rec-1",
        sources=[AudioSource(type="file", channels=[0], source="/data/foo.wav")],
        sampling_rate=16000,
        num_samples=160000,
        duration=10.0,
        num_channels=1,
        checksum="a" * 64,
        custom={"origin": "librispeech", "subset": "train-clean-100"},
    )
    assert rec.checksum == "a" * 64
    assert rec.custom["origin"] == "librispeech"


def test_recording_round_trips_through_json() -> None:
    original = Recording(
        id="rec-1",
        sources=[
            AudioSource(type="file", channels=[0], source="/data/left.wav"),
            AudioSource(type="file", channels=[1], source="/data/right.wav"),
        ],
        sampling_rate=48000,
        num_samples=480000,
        duration=10.0,
        num_channels=2,
    )
    blob = original.model_dump_json()
    restored = Recording.model_validate_json(blob)
    assert restored == original


def test_recording_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Recording(  # type: ignore[call-arg]
            id="rec-1",
            sources=[],
            sampling_rate=16000,
            num_samples=160000,
            num_channels=1,
        )
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_recording.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.recording'`.

- [ ] **Step 3: Write `src/voxkitchen/schema/recording.py`**

```python
"""Recording and AudioSource models.

A Recording describes a physical audio resource — the thing whose bytes live
on disk (or at a URL). A Recording is immutable after ingest: operators that
materialize new audio (format conversion, resampling) create new Recordings
with new ids rather than mutating existing ones.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class AudioSource(BaseModel):
    """One physical source contributing a set of channels to a Recording.

    A Recording can have multiple AudioSources when its audio is split across
    files (e.g. multi-microphone setups where each mic is its own .wav).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["file", "url", "command"]
    channels: list[int]
    source: str


class Recording(BaseModel):
    """A physical audio resource with metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sources: list[AudioSource]
    sampling_rate: int
    num_samples: int
    duration: float
    num_channels: int
    checksum: str | None = None
    custom: dict[str, Any] = {}
```

- [ ] **Step 4: Update `src/voxkitchen/schema/__init__.py` to re-export**

Replace the contents with:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording

__all__ = ["AudioSource", "Provenance", "Recording"]
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_recording.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/schema/recording.py src/voxkitchen/schema/__init__.py tests/unit/schema/test_recording.py
git commit -m "feat(schema): add Recording and AudioSource models"
```

---

## Task 5: Implement Supervision model (TDD)

**Files:**
- Create: `tests/unit/schema/test_supervision.py`
- Create: `src/voxkitchen/schema/supervision.py`
- Modify: `src/voxkitchen/schema/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_supervision.py`:

```python
"""Unit tests for voxkitchen.schema.supervision.Supervision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxkitchen.schema.supervision import Supervision


def test_supervision_minimal_construction() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
    )
    assert sup.id == "sup-1"
    assert sup.recording_id == "rec-1"
    assert sup.start == 0.0
    assert sup.duration == 3.5
    # optional fields default to None / empty
    assert sup.text is None
    assert sup.language is None
    assert sup.speaker is None
    assert sup.gender is None
    assert sup.age_range is None
    assert sup.custom == {}


def test_supervision_with_full_annotations() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        channel=0,
        text="hello world",
        language="en",
        speaker="spk-42",
        gender="f",
        age_range="adult",
        custom={"confidence": 0.97},
    )
    assert sup.text == "hello world"
    assert sup.language == "en"
    assert sup.speaker == "spk-42"
    assert sup.gender == "f"
    assert sup.age_range == "adult"
    assert sup.custom["confidence"] == 0.97


def test_supervision_channel_can_be_list_for_multi_channel_audio() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        channel=[0, 1],
    )
    assert sup.channel == [0, 1]


def test_supervision_rejects_invalid_gender() -> None:
    with pytest.raises(ValidationError):
        Supervision(
            id="sup-1",
            recording_id="rec-1",
            start=0.0,
            duration=3.5,
            gender="x",  # type: ignore[arg-type]
        )


def test_supervision_round_trips_through_json() -> None:
    original = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=1.25,
        duration=2.0,
        text="测试",
        language="zh",
        speaker="spk-1",
    )
    blob = original.model_dump_json()
    restored = Supervision.model_validate_json(blob)
    assert restored == original
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_supervision.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.supervision'`.

- [ ] **Step 3: Write `src/voxkitchen/schema/supervision.py`**

```python
"""Supervision: a labeled time interval over a Recording."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Supervision(BaseModel):
    """A human-interpretable annotation over an interval of a Recording.

    Multiple Supervisions can cover the same Recording (e.g. multiple speakers
    in a conversation) and their time intervals may overlap (e.g. two people
    speaking simultaneously).

    Every annotation field except ``id``, ``recording_id``, ``start``, and
    ``duration`` is optional — operators fill them in progressively as the
    pipeline advances. A Supervision emitted by VAD has no ``text``; after ASR
    it does.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    recording_id: str
    start: float
    duration: float

    channel: int | list[int] | None = None
    text: str | None = None
    language: str | None = None
    speaker: str | None = None
    gender: Literal["m", "f", "o"] | None = None
    age_range: str | None = None
    custom: dict[str, Any] = {}
```

- [ ] **Step 4: Update `src/voxkitchen/schema/__init__.py`**

Replace with:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.schema.supervision import Supervision

__all__ = ["AudioSource", "Provenance", "Recording", "Supervision"]
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_supervision.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/schema/supervision.py src/voxkitchen/schema/__init__.py tests/unit/schema/test_supervision.py
git commit -m "feat(schema): add Supervision model"
```

---

## Task 6: Implement Cut model (TDD)

**Files:**
- Create: `tests/unit/schema/test_cut.py`
- Create: `src/voxkitchen/schema/cut.py`
- Modify: `src/voxkitchen/schema/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_cut.py`:

```python
"""Unit tests for voxkitchen.schema.cut.Cut."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _make_provenance(source_id: str | None = "cut-parent") -> Provenance:
    return Provenance(
        source_cut_id=source_id,
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )


def test_cut_minimal_construction() -> None:
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut.id == "cut-1"
    assert cut.recording_id == "rec-1"
    assert cut.start == 0.0
    assert cut.duration == 3.5
    assert cut.supervisions == []
    assert cut.metrics == {}
    assert cut.custom == {}


def test_cut_with_supervisions_and_metrics() -> None:
    sup = Supervision(
        id="sup-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        text="hello world",
    )
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[sup],
        metrics={"snr": 18.3, "clip_rate": 0.001},
        provenance=_make_provenance(),
    )
    assert len(cut.supervisions) == 1
    assert cut.supervisions[0].text == "hello world"
    assert cut.metrics["snr"] == 18.3


def test_cut_channel_may_be_int_list_or_none() -> None:
    cut_single = Cut(
        id="c1",
        recording_id="r1",
        start=0.0,
        duration=1.0,
        channel=0,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut_single.channel == 0

    cut_multi = Cut(
        id="c2",
        recording_id="r1",
        start=0.0,
        duration=1.0,
        channel=[0, 1],
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut_multi.channel == [0, 1]


def test_cut_rejects_missing_provenance() -> None:
    with pytest.raises(ValidationError):
        Cut(  # type: ignore[call-arg]
            id="cut-1",
            recording_id="rec-1",
            start=0.0,
            duration=3.5,
            supervisions=[],
        )


def test_cut_round_trips_through_json() -> None:
    original = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[
            Supervision(
                id="sup-1",
                recording_id="rec-1",
                start=0.0,
                duration=3.5,
                text="hello world",
                language="en",
            )
        ],
        metrics={"snr": 18.3},
        provenance=_make_provenance(),
        custom={"split": "train"},
    )
    blob = original.model_dump_json()
    restored = Cut.model_validate_json(blob)
    assert restored == original
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_cut.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.cut'`.

- [ ] **Step 3: Write `src/voxkitchen/schema/cut.py`**

```python
"""Cut: a trainable sample referencing a Recording slice with Supervisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


class Cut(BaseModel):
    """A trainable sample — the unit flowing through a pipeline.

    A Cut references a ``[start, start+duration)`` slice of a Recording plus
    the Supervisions that fall within it. Operators transform CutSets by
    producing new Cuts (segmentation creates children, quality filters drop
    Cuts, ASR appends Supervisions) — original Cuts are never mutated.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    recording_id: str
    start: float
    duration: float
    channel: int | list[int] | None = None

    supervisions: list[Supervision]
    metrics: dict[str, float] = {}
    provenance: Provenance
    custom: dict[str, Any] = {}
```

- [ ] **Step 4: Update `src/voxkitchen/schema/__init__.py`**

Replace with:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.schema.supervision import Supervision

__all__ = ["AudioSource", "Cut", "Provenance", "Recording", "Supervision"]
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_cut.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/schema/cut.py src/voxkitchen/schema/__init__.py tests/unit/schema/test_cut.py
git commit -m "feat(schema): add Cut model"
```

---

## Task 7: Implement schema I/O (JSONL.gz with header record)

**Files:**
- Create: `tests/unit/schema/test_io.py`
- Create: `src/voxkitchen/schema/io.py`
- Modify: `src/voxkitchen/schema/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_io.py`:

```python
"""Unit tests for voxkitchen.schema.io: JSONL.gz reading/writing with header."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import (
    SCHEMA_VERSION,
    HeaderRecord,
    IncompatibleSchemaError,
    read_cuts,
    read_header,
    write_cuts,
)
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _make_cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[
            Supervision(id=f"{cid}-sup", recording_id="rec-1", start=0.0, duration=3.5)
        ],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def test_write_then_read_round_trips_all_cuts(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    cuts = [_make_cut(f"cut-{i}") for i in range(3)]
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter(cuts))
    restored = list(read_cuts(path))
    assert restored == cuts


def test_header_is_first_line_of_file(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter([_make_cut("cut-1")]))
    with gzip.open(path, "rt") as f:
        first_line = f.readline()
    parsed = json.loads(first_line)
    assert parsed["__type__"] == "voxkitchen.header"
    assert parsed["schema_version"] == SCHEMA_VERSION


def test_read_header_returns_header_without_consuming_cuts(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="02_vad",
    )
    write_cuts(path, header, iter([_make_cut("cut-1")]))
    read_back = read_header(path)
    assert read_back.stage_name == "02_vad"
    assert read_back.pipeline_run_id == "run-a1b2c3"


def test_incompatible_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "__type__": "voxkitchen.header",
                    "schema_version": "99.0",
                    "created_at": "2026-04-11T10:30:00+00:00",
                    "pipeline_run_id": "run-x",
                    "stage_name": "00_ingest",
                }
            )
            + "\n"
        )
    with pytest.raises(IncompatibleSchemaError):
        list(read_cuts(path))


def test_read_cuts_on_empty_manifest_returns_empty_iterator(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter([]))
    assert list(read_cuts(path)) == []
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_io.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.io'`.

- [ ] **Step 3: Write `src/voxkitchen/schema/io.py`**

```python
"""Serialization of Cuts to and from ``cuts.jsonl.gz`` manifests.

Format (one JSON object per line, gzip-compressed):

    {"__type__": "voxkitchen.header", "schema_version": "0.1", ...}
    {"__type__": "cut", "id": "...", "recording_id": "...", ...}
    {"__type__": "cut", ...}
    ...

The header record enables future schema migrations: readers that encounter
an unknown version can be routed through a migration function.
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.cut import Cut

SCHEMA_VERSION = "0.1"


class IncompatibleSchemaError(RuntimeError):
    """Raised when a manifest's schema version is not understood."""


class HeaderRecord(BaseModel):
    """First-line metadata record in every ``cuts.jsonl.gz`` file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    created_at: datetime
    pipeline_run_id: str
    stage_name: str


def write_cuts(path: Path, header: HeaderRecord, cuts: Iterable[Cut]) -> None:
    """Write a header followed by a stream of Cuts to a gzipped JSONL file.

    Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        header_obj = {"__type__": "voxkitchen.header", **header.model_dump(mode="json")}
        f.write(json.dumps(header_obj, ensure_ascii=False) + "\n")
        for cut in cuts:
            cut_obj = {"__type__": "cut", **cut.model_dump(mode="json")}
            f.write(json.dumps(cut_obj, ensure_ascii=False) + "\n")


def read_header(path: Path) -> HeaderRecord:
    """Return the header record of a manifest without reading the rest."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline()
    if not first:
        raise IncompatibleSchemaError(f"manifest is empty: {path}")
    parsed = json.loads(first)
    if parsed.get("__type__") != "voxkitchen.header":
        raise IncompatibleSchemaError(
            f"expected header on first line of {path}, got {parsed.get('__type__')!r}"
        )
    payload = {k: v for k, v in parsed.items() if k != "__type__"}
    return HeaderRecord.model_validate(payload)


def read_cuts(path: Path) -> Iterator[Cut]:
    """Yield Cuts from a manifest, validating the header first.

    Raises ``IncompatibleSchemaError`` if the file has no header or the
    schema version does not match ``SCHEMA_VERSION``.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        parsed = json.loads(first)
        if parsed.get("__type__") != "voxkitchen.header":
            raise IncompatibleSchemaError(
                f"expected header on first line of {path}, got {parsed.get('__type__')!r}"
            )
        if parsed.get("schema_version") != SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"schema version mismatch in {path}: "
                f"file is {parsed.get('schema_version')!r}, "
                f"reader supports {SCHEMA_VERSION!r}"
            )
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("__type__") != "cut":
                continue
            cut_payload = {k: v for k, v in obj.items() if k != "__type__"}
            yield Cut.model_validate(cut_payload)
```

- [ ] **Step 4: Update `src/voxkitchen/schema/__init__.py`**

Replace with:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import (
    SCHEMA_VERSION,
    HeaderRecord,
    IncompatibleSchemaError,
    read_cuts,
    read_header,
    write_cuts,
)
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.schema.supervision import Supervision

__all__ = [
    "SCHEMA_VERSION",
    "AudioSource",
    "Cut",
    "HeaderRecord",
    "IncompatibleSchemaError",
    "Provenance",
    "Recording",
    "Supervision",
    "read_cuts",
    "read_header",
    "write_cuts",
]
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_io.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/schema/io.py src/voxkitchen/schema/__init__.py tests/unit/schema/test_io.py
git commit -m "feat(schema): add JSONL.gz I/O with header and schema version"
```

---

## Task 8: Implement CutSet (in-memory operations)

**Files:**
- Create: `tests/unit/schema/test_cutset.py`
- Create: `src/voxkitchen/schema/cutset.py`
- Modify: `src/voxkitchen/schema/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/schema/test_cutset.py`:

```python
"""Unit tests for voxkitchen.schema.cutset.CutSet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _cut(cid: str, duration: float = 1.0) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=duration,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def _header(stage: str = "00_ingest") -> HeaderRecord:
    return HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name=stage,
    )


def test_cutset_len_and_iter() -> None:
    cs = CutSet([_cut("c0"), _cut("c1"), _cut("c2")])
    assert len(cs) == 3
    ids = [c.id for c in cs]
    assert ids == ["c0", "c1", "c2"]


def test_cutset_split_into_n_shards_balanced() -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(10)])
    shards = cs.split(3)
    assert len(shards) == 3
    # No Cut lost, no duplication
    all_ids = [c.id for shard in shards for c in shard]
    assert sorted(all_ids) == sorted(c.id for c in cs)
    # Sizes within 1 of each other
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_cutset_split_n_larger_than_len_produces_empty_shards() -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])
    shards = cs.split(5)
    assert len(shards) == 5
    assert sum(len(s) for s in shards) == 2


def test_cutset_filter_preserves_matching_cuts_only() -> None:
    cs = CutSet([_cut("c0", 1.0), _cut("c1", 5.0), _cut("c2", 3.0)])
    filtered = cs.filter(lambda c: c.duration >= 3.0)
    assert [c.id for c in filtered] == ["c1", "c2"]


def test_cutset_map_applies_transformation() -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])

    def add_metric(c: Cut) -> Cut:
        return c.model_copy(update={"metrics": {"snr": 20.0}})

    mapped = cs.map(add_metric)
    assert all(c.metrics["snr"] == 20.0 for c in mapped)


def test_cutset_merge_concatenates_all_cuts() -> None:
    a = CutSet([_cut("a0"), _cut("a1")])
    b = CutSet([_cut("b0")])
    c = CutSet([_cut("c0"), _cut("c1")])
    merged = CutSet.merge([a, b, c])
    assert [x.id for x in merged] == ["a0", "a1", "b0", "c0", "c1"]


def test_cutset_to_and_from_jsonl_gz_round_trips(tmp_path: Path) -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])
    path = tmp_path / "cuts.jsonl.gz"
    cs.to_jsonl_gz(path, _header())
    restored = CutSet.from_jsonl_gz(path)
    assert [c.id for c in restored] == ["c0", "c1"]


def test_cutset_concat_from_disk_joins_shards_in_order(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"shard_{i}.jsonl.gz"
        CutSet([_cut(f"s{i}c0"), _cut(f"s{i}c1")]).to_jsonl_gz(p, _header())
        paths.append(p)
    merged = CutSet.concat_from_disk(paths)
    assert [c.id for c in merged] == [
        "s0c0",
        "s0c1",
        "s1c0",
        "s1c1",
        "s2c0",
        "s2c1",
    ]
```

- [ ] **Step 2: Run the test and verify it fails**

```bash
pytest tests/unit/schema/test_cutset.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.schema.cutset'`.

- [ ] **Step 3: Write `src/voxkitchen/schema/cutset.py`**

```python
"""CutSet: a lazy, functional wrapper over a sequence of Cuts.

CutSet is deliberately not a Pydantic model. It is a thin layer above an
iterable of Cuts backed by either an in-memory list or a JSONL.gz file on
disk. Plan 1 implements the in-memory path; lazy streaming I/O (opening a
file and yielding Cuts without reading the whole thing into memory) is
also supported through ``from_jsonl_gz`` since ``read_cuts`` already returns
an iterator.

The key operations needed by the pipeline engine are:
- ``split(n)``: shard into N CutSets for GPU/CPU pool workers
- ``filter(pred)``: drop Cuts that don't match
- ``map(fn)``: single-Cut transformation
- ``merge(cutsets)``: concatenate several CutSets
- ``to_jsonl_gz(path, header)``: persist to disk
- ``from_jsonl_gz(path)``: load from disk (validates header + schema version)
- ``concat_from_disk(paths)``: lazy concatenation of shard files (used by
  GpuPoolExecutor after workers finish)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import HeaderRecord, read_cuts, write_cuts


class CutSet:
    """A sequence of Cuts with functional operations and disk I/O helpers."""

    def __init__(self, cuts: Iterable[Cut]) -> None:
        # Materialize to a list. Lazy-streaming variants can be added later
        # if memory becomes a concern; for Plan 1 we keep the semantics simple
        # and predictable.
        self._cuts: list[Cut] = list(cuts)

    def __len__(self) -> int:
        return len(self._cuts)

    def __iter__(self) -> Iterator[Cut]:
        return iter(self._cuts)

    def split(self, n: int) -> list[CutSet]:
        """Split into ``n`` roughly-equal CutSets.

        When ``n > len(self)``, trailing CutSets may be empty. Preserves order.
        """
        if n <= 0:
            raise ValueError(f"split n must be positive, got {n}")
        shards: list[list[Cut]] = [[] for _ in range(n)]
        for i, cut in enumerate(self._cuts):
            shards[i % n].append(cut)
        return [CutSet(s) for s in shards]

    def filter(self, predicate: Callable[[Cut], bool]) -> CutSet:
        return CutSet(c for c in self._cuts if predicate(c))

    def map(self, fn: Callable[[Cut], Cut]) -> CutSet:
        return CutSet(fn(c) for c in self._cuts)

    @classmethod
    def merge(cls, cutsets: Iterable[CutSet]) -> CutSet:
        """Concatenate multiple CutSets into one, preserving order."""
        out: list[Cut] = []
        for cs in cutsets:
            out.extend(cs)
        return cls(out)

    def to_jsonl_gz(self, path: Path, header: HeaderRecord) -> None:
        write_cuts(path, header, iter(self._cuts))

    @classmethod
    def from_jsonl_gz(cls, path: Path) -> CutSet:
        return cls(read_cuts(path))

    @classmethod
    def concat_from_disk(cls, paths: Sequence[Path]) -> CutSet:
        """Read and concatenate multiple manifest files in the given order."""
        out: list[Cut] = []
        for p in paths:
            out.extend(read_cuts(p))
        return cls(out)
```

- [ ] **Step 4: Update `src/voxkitchen/schema/__init__.py`**

Replace with:

```python
"""VoxKitchen data schema: Recording, Supervision, Cut, CutSet, Provenance."""

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import (
    SCHEMA_VERSION,
    HeaderRecord,
    IncompatibleSchemaError,
    read_cuts,
    read_header,
    write_cuts,
)
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.schema.supervision import Supervision

__all__ = [
    "SCHEMA_VERSION",
    "AudioSource",
    "Cut",
    "CutSet",
    "HeaderRecord",
    "IncompatibleSchemaError",
    "Provenance",
    "Recording",
    "Supervision",
    "read_cuts",
    "read_header",
    "write_cuts",
]
```

- [ ] **Step 5: Run the test and verify it passes**

```bash
pytest tests/unit/schema/test_cutset.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/schema/cutset.py src/voxkitchen/schema/__init__.py tests/unit/schema/test_cutset.py
git commit -m "feat(schema): add CutSet with split, filter, map, merge, and I/O"
```

---

## Task 9: Run full test suite and lint

**Files:**
- None (verification task)

- [ ] **Step 1: Run the entire test suite**

```bash
pytest -v
```

Expected: all tests from Tasks 2–8 pass. Approximate count:
- `test_cli_smoke.py`: 2
- `test_provenance.py`: 5
- `test_recording.py`: 7
- `test_supervision.py`: 5
- `test_cut.py`: 5
- `test_io.py`: 5
- `test_cutset.py`: 8

Total: **37 passed**. Run time should be well under 10 seconds.

- [ ] **Step 2: Run ruff lint**

```bash
ruff check src tests
```

Expected: zero errors. If there are any, fix them and re-run. Common issues to fix:
- unused imports
- unsorted imports
- missing trailing newline

- [ ] **Step 3: Run ruff format**

```bash
ruff format src tests
```

Expected: `X files reformatted` or `X files left unchanged`. If it reformats files, inspect the diff and commit.

- [ ] **Step 4: Run mypy strict type check**

```bash
mypy src/voxkitchen tests
```

Expected: `Success: no issues found in N source files`. If mypy complains, fix the issues — do not add `# type: ignore` unless the issue is in third-party code.

- [ ] **Step 5: Commit any lint/format fixes**

If ruff or mypy fixes were needed:

```bash
git add src tests
git commit -m "style: ruff format and mypy fixes"
```

Otherwise skip this step.

---

## Task 10: Set up pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.10
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: [--maxkb=500]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.5
          - types-PyYAML
        args: [--strict]
        files: ^(src/voxkitchen|tests)/
```

- [ ] **Step 2: Install the hooks and run once**

```bash
pre-commit install
pre-commit run --all-files
```

Expected: all hooks pass on the existing code. If any fail, inspect and fix — most common failures are whitespace/EOF nits that pre-commit auto-fixes on its own.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: add pre-commit hook config for ruff and mypy"
```

---

## Task 11: Add GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # hatch-vcs needs tags

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip

      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Ruff lint
        run: ruff check src tests

      - name: Ruff format check
        run: ruff format --check src tests

      - name: Mypy
        run: mypy src/voxkitchen tests

      - name: Pytest
        run: pytest -v --cov=voxkitchen --cov-report=term-missing
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for lint, type, and tests"
```

---

## Task 12: Final verification and tag

**Files:**
- None (verification task)

- [ ] **Step 1: Run the full suite one final time**

```bash
pytest -v
ruff check src tests
ruff format --check src tests
mypy src/voxkitchen tests
```

Expected: everything green.

- [ ] **Step 2: Verify `vkit --help` still works**

```bash
vkit --help
```

Expected: shows the 6 placeholder commands with helpful descriptions.

- [ ] **Step 3: Verify all schema types are importable from the public API**

```bash
python -c "from voxkitchen.schema import AudioSource, Cut, CutSet, HeaderRecord, IncompatibleSchemaError, Provenance, Recording, SCHEMA_VERSION, Supervision, read_cuts, read_header, write_cuts; print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Verify git log shows ~11 commits**

```bash
git log --oneline
```

Expected (order may vary):
```
<hash> feat(schema): add CutSet with split, filter, map, merge, and I/O
<hash> feat(schema): add JSONL.gz I/O with header and schema version
<hash> feat(schema): add Cut model
<hash> feat(schema): add Supervision model
<hash> feat(schema): add Recording and AudioSource models
<hash> feat(schema): add Provenance model with unit tests
<hash> feat: add src layout, placeholder CLI, and smoke test
<hash> chore: bootstrap voxkitchen project with metadata and spec
<hash> chore: add pre-commit hook config for ruff and mypy
<hash> ci: add GitHub Actions workflow for lint, type, and tests
<hash> (optional) style: ruff format and mypy fixes
```

- [ ] **Step 5: Tag Plan 1 completion**

```bash
git tag -a plan-01-foundation -m "Plan 1 complete: project skeleton + data schema"
```

Plan 1 is done. Plan 2 (Pipeline Engine) is the next step — it builds on top of `voxkitchen.schema` without modifying any of the files produced here.

---

## Plan 1 Completion Checklist

Before declaring Plan 1 complete, verify every item:

- [ ] `pip install -e ".[dev]"` succeeds in a clean venv on Python 3.10, 3.11, and 3.12
- [ ] `vkit --help` runs and lists 6 commands (init, ingest, validate, run, inspect, viz)
- [ ] All 37+ unit tests pass in under 10 seconds
- [ ] `ruff check src tests` reports zero errors
- [ ] `ruff format --check src tests` reports zero changes needed
- [ ] `mypy src/voxkitchen tests` reports no issues (strict mode)
- [ ] `from voxkitchen.schema import AudioSource, Cut, CutSet, Provenance, Recording, Supervision` succeeds
- [ ] `CutSet([...]).to_jsonl_gz(path, header)` followed by `CutSet.from_jsonl_gz(path)` round-trips cuts exactly
- [ ] Every schema file is ≤ 150 lines (signals good isolation)
- [ ] Every file ends with a newline
- [ ] Every Python file has `from __future__ import annotations` at the top (except `__init__.py`)
- [ ] Git history has one commit per logical task — no squashed or mixed commits
- [ ] `.github/workflows/ci.yml` is present and uses the 3.10/3.11/3.12 matrix
- [ ] `.pre-commit-config.yaml` is installed (`pre-commit install`)
