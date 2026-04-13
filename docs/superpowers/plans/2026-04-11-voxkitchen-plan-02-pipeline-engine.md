# VoxKitchen Plan 2: Pipeline Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the VoxKitchen pipeline engine: operator abstraction + registry, declarative YAML pipeline specs, a runner that orchestrates stages end-to-end with checkpoint/resume and aggressive GC, and CPU/GPU pool executors. Wire the real `vkit run` and `vkit validate` commands into the CLI.

**Architecture:** Strictly layered — `operators/` depends only on `schema/`; `pipeline/` depends on `schema/` + `operators/`; `cli/` depends on all. Runner is the orchestration root and is constructor-injected with executors to keep it testable. An `IdentityOperator` (no-op) ships with Plan 2 to enable end-to-end integration tests without requiring any heavy dependencies. A `ManifestIngestSource` reads a pre-built `cuts.jsonl.gz` — it doubles as both (a) a real user-facing ingest option and (b) the integration test scaffolding for Plan 2. All other ingest sources (directory scan, recipes) land in Plan 3.

**Tech Stack:** Python 3.10+, Pydantic v2, PyYAML, multiprocessing (spawn context), Typer. No new core deps are introduced.

**Spec reference:** `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/docs/superpowers/specs/2026-04-11-voxkitchen-design.md` — sections 2 (repo skeleton), 4 (operator abstraction), 5 (pipeline & runner), 6 (CLI).

**Prior art:** `/Users/mobvoi/Downloads/USC/1/SpeechDatasetHub/docs/superpowers/plans/2026-04-11-voxkitchen-plan-01-foundation.md` (Plan 1 produced `voxkitchen.schema` — Recording, Supervision, Cut, CutSet, Provenance, JSONL.gz I/O with header + schema versioning).

---

## File Structure Produced by This Plan

```
src/voxkitchen/
├── operators/
│   ├── __init__.py                       # re-exports Operator, OperatorConfig, register_operator, get_operator, ...
│   ├── base.py                           # Operator ABC + OperatorConfig
│   ├── registry.py                       # _REGISTRY, register_operator, get_operator, UnknownOperatorError, MissingExtrasError
│   └── noop/
│       ├── __init__.py
│       └── identity.py                   # IdentityOperator: CutSet → CutSet unchanged (testing scaffold)
│
├── pipeline/
│   ├── __init__.py
│   ├── spec.py                           # PipelineSpec, StageSpec, IngestSpec
│   ├── loader.py                         # load_pipeline_spec(path) + ${name} / ${run_id} / ${env:VAR} interpolation
│   ├── context.py                        # RunContext dataclass
│   ├── checkpoint.py                     # _SUCCESS marker helpers, resume detection
│   ├── gc.py                             # GcPlan, compute_gc_plan, run_gc, empty_trash
│   ├── executor.py                       # Executor Protocol + CpuPoolExecutor + GpuPoolExecutor
│   └── runner.py                         # run_pipeline() — main orchestration loop
│
├── ingest/
│   ├── __init__.py                       # NEW in Plan 2: just the manifest source
│   ├── base.py                           # IngestSource ABC
│   └── manifest_import.py                # ManifestIngestSource: reads an existing cuts.jsonl.gz
│
├── cli/
│   ├── main.py                           # MODIFIED: init/ingest/viz still placeholders; validate/run wired to real impls
│   ├── validate.py                       # NEW: real `vkit validate`
│   └── run.py                            # NEW: real `vkit run`
│
└── utils/
    └── run_id.py                         # NEW: generate_run_id() helper (ULID-ish timestamp + random suffix)

tests/
├── unit/
│   ├── operators/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_registry.py
│   │   └── test_identity.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── test_spec.py
│   │   ├── test_loader.py
│   │   ├── test_context.py
│   │   ├── test_checkpoint.py
│   │   ├── test_gc.py
│   │   ├── test_executor_cpu.py
│   │   ├── test_executor_gpu.py
│   │   └── test_runner.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── test_manifest_import.py
│   └── cli/
│       ├── __init__.py
│       ├── test_validate.py
│       └── test_run.py
│
└── integration/
    ├── __init__.py
    ├── conftest.py                       # shared fixtures: tmp_pipeline, sample cutset
    └── test_pipeline_e2e.py              # full YAML → run → verify stages
```

**Responsibility per file:**

- `operators/base.py` — `Operator` ABC (with `name`, `config_cls`, `device`, `produces_audio`, `reads_audio_bytes`, `required_extras`, `setup`, `process`, `teardown`) and `OperatorConfig` Pydantic base. Defines the contract every operator implements.
- `operators/registry.py` — process-global `_REGISTRY`, `register_operator` decorator, `get_operator(name)`, `list_operators()`, `UnknownOperatorError`, `MissingExtrasError`.
- `operators/noop/identity.py` — `IdentityOperator`, the simplest possible operator. Produces no new audio. Used as the default test scaffold.
- `pipeline/spec.py` — Pydantic models for pipeline YAML.
- `pipeline/loader.py` — YAML → `PipelineSpec` with `${name}` / `${run_id}` / `${env:VAR}` string interpolation.
- `pipeline/context.py` — `RunContext` dataclass: work_dir, run_id, stage_index, stage_name, num_gpus, num_cpu_workers, gc_mode, device. Pickle-friendly (no logger, no open file handles).
- `pipeline/checkpoint.py` — `write_success_marker`, `is_stage_complete`, `find_last_completed_stage`, `stage_dir_name` helper.
- `pipeline/gc.py` — `GcPlan`, `compute_gc_plan`, `run_gc`, `empty_trash`.
- `pipeline/executor.py` — `Executor` Protocol; `CpuPoolExecutor` (multiprocessing.Pool with spawn context); `GpuPoolExecutor` (N subprocesses, one per GPU, sets `CUDA_VISIBLE_DEVICES`).
- `pipeline/runner.py` — `run_pipeline(spec, ...)` — the orchestration loop. Constructor-injected executors for testability.
- `ingest/base.py` — `IngestSource` ABC with `run(ctx) -> tuple[CutSet, list[Recording]]`.
- `ingest/manifest_import.py` — `ManifestIngestSource` implementing the `source: manifest` YAML option by reading an existing `cuts.jsonl.gz`.
- `utils/run_id.py` — `generate_run_id()` returns a short sortable string like `run-20260411T103000-a1b2`.
- `cli/validate.py` / `cli/run.py` — real implementations replacing Plan 1 placeholders.

**Layering rules (MUST hold after Plan 2):**

- `schema/` depends on nothing internal (Plan 1 invariant).
- `operators/` depends on `schema/` + `utils/`. Must NOT import `pipeline/` at module level. `RunContext` is imported only under `TYPE_CHECKING` to avoid circularity (operators use it but don't construct it).
- `ingest/` depends on `schema/` + `utils/`. Same `TYPE_CHECKING` rule for `RunContext`.
- `pipeline/` depends on `schema/` + `operators/` + `ingest/` + `utils/`.
- `cli/` depends on all the above.
- No module imports `cli/`.

---

## Task 1: Bootstrap pipeline, operators, ingest packages

**Files:**
- Create: `src/voxkitchen/operators/__init__.py`
- Create: `src/voxkitchen/operators/noop/__init__.py`
- Create: `src/voxkitchen/pipeline/__init__.py`
- Create: `src/voxkitchen/ingest/__init__.py`
- Create: `src/voxkitchen/utils/run_id.py`
- Create: `tests/unit/operators/__init__.py`
- Create: `tests/unit/operators/noop/__init__.py` (if needed)
- Create: `tests/unit/pipeline/__init__.py`
- Create: `tests/unit/ingest/__init__.py`
- Create: `tests/unit/cli/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/conftest.py`
- Create: `tests/unit/utils/test_run_id.py`

- [ ] **Step 1: Activate venv and create directories**

```bash
cd /Users/mobvoi/Downloads/USC/1/SpeechDatasetHub
source .venv/bin/activate
mkdir -p src/voxkitchen/operators/noop
mkdir -p src/voxkitchen/pipeline
mkdir -p src/voxkitchen/ingest
mkdir -p tests/unit/operators/noop
mkdir -p tests/unit/pipeline
mkdir -p tests/unit/ingest
mkdir -p tests/unit/cli
mkdir -p tests/unit/utils
mkdir -p tests/integration
```

- [ ] **Step 2: Write empty `__init__.py` stubs**

Create each of these as one-line docstring files:

`src/voxkitchen/operators/__init__.py`:
```python
"""VoxKitchen operators: transformations from CutSet to CutSet."""
```

`src/voxkitchen/operators/noop/__init__.py`:
```python
"""No-op operators used as test scaffolding."""
```

`src/voxkitchen/pipeline/__init__.py`:
```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""
```

`src/voxkitchen/ingest/__init__.py`:
```python
"""VoxKitchen ingest sources: how Cuts enter a pipeline."""
```

`tests/unit/operators/__init__.py`, `tests/unit/operators/noop/__init__.py`, `tests/unit/pipeline/__init__.py`, `tests/unit/ingest/__init__.py`, `tests/unit/cli/__init__.py`, `tests/unit/utils/__init__.py`, `tests/integration/__init__.py` — all empty files.

- [ ] **Step 3: Write `tests/unit/utils/test_run_id.py` (failing)**

```python
"""Unit tests for voxkitchen.utils.run_id.generate_run_id."""

from __future__ import annotations

import re

from voxkitchen.utils.run_id import generate_run_id


def test_run_id_is_nonempty_string() -> None:
    rid = generate_run_id()
    assert isinstance(rid, str)
    assert len(rid) > 0


def test_run_id_has_prefix_and_sortable_timestamp() -> None:
    """Format: run-YYYYMMDDTHHMMSS-<4-hex-chars>"""
    rid = generate_run_id()
    assert re.fullmatch(r"run-\d{8}T\d{6}-[0-9a-f]{4}", rid) is not None


def test_run_ids_are_unique_across_calls() -> None:
    ids = {generate_run_id() for _ in range(50)}
    assert len(ids) == 50
```

- [ ] **Step 4: Run the test, see it fail**

```bash
pytest tests/unit/utils/test_run_id.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.utils.run_id'`.

- [ ] **Step 5: Write `src/voxkitchen/utils/run_id.py`**

```python
"""Generate short, sortable, deterministically-formatted pipeline run IDs."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone


def generate_run_id() -> str:
    """Return a fresh run id like ``run-20260411T103000-a1b2``.

    Format:
    - ``run-`` prefix for greppability
    - Compact ISO-8601 UTC timestamp (no punctuation) for sortability
    - 4-hex-character random suffix to disambiguate runs within the same second
    """
    now = datetime.now(tz=timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    suffix = secrets.token_hex(2)
    return f"run-{ts}-{suffix}"
```

- [ ] **Step 6: Run the test, see it pass**

```bash
pytest tests/unit/utils/test_run_id.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Write `tests/integration/conftest.py`**

```python
"""Shared fixtures for VoxKitchen integration tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance


def _make_cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="fixture-run",
        ),
    )


@pytest.fixture
def sample_cutset() -> CutSet:
    """A deterministic 5-cut CutSet for integration tests."""
    return CutSet([_make_cut(f"c{i}") for i in range(5)])


@pytest.fixture
def sample_manifest_path(tmp_path: Path, sample_cutset: CutSet) -> Path:
    """Write sample_cutset to a manifest on disk and return the path."""
    path = tmp_path / "input_cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture-run",
        stage_name="00_ingest",
    )
    sample_cutset.to_jsonl_gz(path, header)
    return path
```

- [ ] **Step 8: Verify linting and type checks are clean**

```bash
ruff check src tests
ruff format --check src tests
mypy src/voxkitchen tests
```

All three must be clean. Fix with `ruff format` if needed.

- [ ] **Step 9: Commit**

```bash
git add src/voxkitchen/operators src/voxkitchen/pipeline src/voxkitchen/ingest src/voxkitchen/utils/run_id.py tests/unit/operators tests/unit/pipeline tests/unit/ingest tests/unit/cli tests/unit/utils tests/integration
git commit -m "chore: scaffold pipeline/operators/ingest packages and run_id helper"
```

---

## Task 2: Operator base class + OperatorConfig (TDD)

**Files:**
- Create: `tests/unit/operators/test_base.py`
- Create: `src/voxkitchen/operators/base.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/operators/test_base.py`:

```python
"""Unit tests for voxkitchen.operators.base."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class DemoConfig(OperatorConfig):
    threshold: float = 0.5


class DemoOperator(Operator):
    name = "demo"
    config_cls = DemoConfig

    def process(self, cuts: CutSet) -> CutSet:
        return cuts


def test_operator_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DemoConfig.model_validate({"threshold": 0.5, "surprise": "boom"})


def test_operator_config_applies_defaults() -> None:
    cfg = DemoConfig()
    assert cfg.threshold == 0.5


def test_operator_instantiates_with_config_and_ctx() -> None:
    cfg = DemoConfig(threshold=0.9)
    # ctx is not used at construction time; pass a sentinel
    op = DemoOperator(cfg, ctx=object())  # type: ignore[arg-type]
    assert op.config.threshold == 0.9
    assert op.ctx is not None


def test_operator_setup_and_teardown_are_noops_by_default() -> None:
    cfg = DemoConfig()
    op = DemoOperator(cfg, ctx=object())  # type: ignore[arg-type]
    # should not raise
    op.setup()
    op.teardown()


def test_operator_class_attributes_have_sensible_defaults() -> None:
    assert DemoOperator.device == "cpu"
    assert DemoOperator.produces_audio is False
    assert DemoOperator.reads_audio_bytes is True
    assert DemoOperator.required_extras == []


def test_abstract_process_enforced_by_abc() -> None:
    class IncompleteOperator(Operator):
        name = "incomplete"
        config_cls = DemoConfig

    with pytest.raises(TypeError):
        IncompleteOperator(DemoConfig(), ctx=object())  # type: ignore[abstract,arg-type]
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/operators/test_base.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.operators.base'`.

- [ ] **Step 3: Write `src/voxkitchen/operators/base.py`**

```python
"""Operator ABC and OperatorConfig base.

Every VoxKitchen operator subclasses ``Operator`` and declares:

1. A ``name`` (string, unique across the registry).
2. A ``config_cls`` subclass of ``OperatorConfig`` describing its parameters.
3. A ``process(cuts) -> cuts`` implementation.

Operators may override ``setup()`` and ``teardown()`` for model loading/release.
They may also override the class variables ``device`` (``"cpu"`` | ``"gpu"``),
``produces_audio`` (whether the operator creates new audio files on disk),
``reads_audio_bytes`` (whether downstream stages need to read audio samples
from this stage's outputs), and ``required_extras`` (names of pyproject
optional-dependencies groups this operator needs).

``RunContext`` is imported only under ``TYPE_CHECKING`` so that ``operators/``
does not depend on ``pipeline/`` at import time — this preserves the one-way
layering rule (``operators → schema``, ``pipeline → operators + schema``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class OperatorConfig(BaseModel):
    """Base class for operator parameter models. Forbids unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Operator(ABC):
    """Base class for all VoxKitchen operators.

    Subclasses MUST set the class-level ``name`` and ``config_cls``; the ABC
    does not enforce this at import time, but ``register_operator`` in
    ``registry.py`` rejects classes that are missing either.
    """

    # Identity and configuration
    name: ClassVar[str] = ""  # overridden by subclasses
    config_cls: ClassVar[type[OperatorConfig]] = OperatorConfig

    # Execution metadata (defaults suitable for most pure-Python CPU operators)
    device: ClassVar[Literal["cpu", "gpu"]] = "cpu"
    produces_audio: ClassVar[bool] = False
    reads_audio_bytes: ClassVar[bool] = True
    required_extras: ClassVar[list[str]] = []

    def __init__(self, config: OperatorConfig, ctx: "RunContext") -> None:
        self.config = config
        self.ctx = ctx

    def setup(self) -> None:
        """Called once per worker process before ``process`` is invoked.

        Override to load models, warm caches, or allocate GPU memory.
        """
        return None

    @abstractmethod
    def process(self, cuts: CutSet) -> CutSet:
        """Transform a CutSet into a new CutSet. Must be implemented."""

    def teardown(self) -> None:
        """Called once per worker process before the worker exits.

        Override to release GPU memory, close file handles, etc.
        """
        return None
```

- [ ] **Step 4: Update `src/voxkitchen/operators/__init__.py`**

```python
"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig

__all__ = ["Operator", "OperatorConfig"]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/operators/test_base.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Run lint/type/full suite**

```bash
ruff check src tests
mypy src/voxkitchen tests
pytest -q
```

All green.

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/operators/base.py src/voxkitchen/operators/__init__.py tests/unit/operators/test_base.py
git commit -m "feat(operators): add Operator ABC and OperatorConfig base"
```

---

## Task 3: Operator registry (TDD)

**Files:**
- Create: `tests/unit/operators/test_registry.py`
- Create: `src/voxkitchen/operators/registry.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/operators/test_registry.py`:

```python
"""Unit tests for voxkitchen.operators.registry."""

from __future__ import annotations

import pytest

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    _REGISTRY,
    get_operator,
    list_operators,
    register_operator,
)
from voxkitchen.schema.cutset import CutSet


class _TestConfig(OperatorConfig):
    x: int = 0


@pytest.fixture(autouse=True)
def _clear_registry() -> None:
    """Each test runs against a clean registry snapshot."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _make_op(op_name: str) -> type[Operator]:
    class _DynOp(Operator):
        name = op_name
        config_cls = _TestConfig

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    return _DynOp


def test_register_operator_adds_to_registry() -> None:
    op_cls = _make_op("alpha")
    register_operator(op_cls)
    assert "alpha" in _REGISTRY
    assert _REGISTRY["alpha"] is op_cls


def test_register_operator_returns_the_class_for_decorator_use() -> None:
    op_cls = _make_op("beta")
    returned = register_operator(op_cls)
    assert returned is op_cls


def test_register_operator_rejects_duplicate_names() -> None:
    register_operator(_make_op("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_operator(_make_op("dup"))


def test_register_operator_rejects_empty_name() -> None:
    class _Empty(Operator):
        name = ""
        config_cls = _TestConfig

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    with pytest.raises(ValueError, match="must declare"):
        register_operator(_Empty)


def test_get_operator_returns_registered_class() -> None:
    op_cls = _make_op("gamma")
    register_operator(op_cls)
    assert get_operator("gamma") is op_cls


def test_get_operator_raises_on_unknown_name() -> None:
    register_operator(_make_op("delta"))
    with pytest.raises(UnknownOperatorError) as exc_info:
        get_operator("deltaa")  # typo
    # Suggestion should include the close match
    assert "delta" in exc_info.value.suggestions


def test_list_operators_returns_sorted_names() -> None:
    for n in ["zulu", "alpha", "mike"]:
        register_operator(_make_op(n))
    assert list_operators() == ["alpha", "mike", "zulu"]


def test_missing_extras_error_message_includes_pip_hint() -> None:
    err = MissingExtrasError("faster_whisper_asr", ["asr"])
    msg = str(err)
    assert "faster_whisper_asr" in msg
    assert "pip install voxkitchen[asr]" in msg
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/operators/test_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'voxkitchen.operators.registry'`.

- [ ] **Step 3: Write `src/voxkitchen/operators/registry.py`**

```python
"""Process-global operator registry.

Plan 2 ships a manual registry populated via ``register_operator``. Plan 8
will layer entry-points discovery on top via ``voxkitchen.plugins.discovery``
without changing the API here.
"""

from __future__ import annotations

import difflib

from voxkitchen.operators.base import Operator

_REGISTRY: dict[str, type[Operator]] = {}


class UnknownOperatorError(KeyError):
    """Raised when an operator name is not in the registry."""

    def __init__(self, name: str, suggestions: list[str]) -> None:
        self.op_name = name
        self.suggestions = suggestions
        hint = f"; did you mean {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(f"unknown operator: {name!r}{hint}")


class MissingExtrasError(ImportError):
    """Raised when an operator's required extras are not installed."""

    def __init__(self, op_name: str, extras: list[str]) -> None:
        self.op_name = op_name
        self.extras = extras
        extras_str = ",".join(extras)
        super().__init__(
            f"operator {op_name!r} requires extras not installed. "
            f"Install with: pip install voxkitchen[{extras_str}]"
        )


def register_operator(op_cls: type[Operator]) -> type[Operator]:
    """Register an Operator subclass in the global registry.

    Used as a decorator:

        @register_operator
        class MyOp(Operator):
            ...

    Raises ``ValueError`` if the class does not declare a non-empty ``name`` or
    if an operator with the same name is already registered.
    """
    if not getattr(op_cls, "name", ""):
        raise ValueError(
            f"{op_cls.__name__} must declare a non-empty class-level 'name' attribute"
        )
    if op_cls.name in _REGISTRY:
        existing = _REGISTRY[op_cls.name]
        raise ValueError(
            f"operator {op_cls.name!r} already registered "
            f"(existing: {existing.__module__}.{existing.__name__}, "
            f"new: {op_cls.__module__}.{op_cls.__name__})"
        )
    _REGISTRY[op_cls.name] = op_cls
    return op_cls


def get_operator(name: str) -> type[Operator]:
    """Return the Operator subclass registered under ``name``.

    Raises ``UnknownOperatorError`` with fuzzy-match suggestions if not found.
    """
    if name in _REGISTRY:
        return _REGISTRY[name]
    suggestions = difflib.get_close_matches(name, list(_REGISTRY.keys()), n=3, cutoff=0.6)
    raise UnknownOperatorError(name, suggestions)


def list_operators() -> list[str]:
    """Return all registered operator names, sorted."""
    return sorted(_REGISTRY.keys())
```

- [ ] **Step 4: Update `src/voxkitchen/operators/__init__.py`**

```python
"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
    list_operators,
    register_operator,
)

__all__ = [
    "MissingExtrasError",
    "Operator",
    "OperatorConfig",
    "UnknownOperatorError",
    "get_operator",
    "list_operators",
    "register_operator",
]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/operators/test_registry.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/operators/registry.py src/voxkitchen/operators/__init__.py tests/unit/operators/test_registry.py
git commit -m "feat(operators): add registry with fuzzy-match suggestions"
```

---

## Task 4: IdentityOperator (TDD)

**Files:**
- Create: `tests/unit/operators/test_identity.py`
- Create: `src/voxkitchen/operators/noop/identity.py`
- Modify: `src/voxkitchen/operators/noop/__init__.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/operators/test_identity.py`:

```python
"""Unit tests for voxkitchen.operators.noop.identity.IdentityOperator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.noop.identity import IdentityConfig, IdentityOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def test_identity_operator_is_registered_as_identity() -> None:
    assert get_operator("identity") is IdentityOperator


def test_identity_operator_is_cpu_and_noop() -> None:
    assert IdentityOperator.device == "cpu"
    assert IdentityOperator.produces_audio is False
    assert IdentityOperator.required_extras == []


def test_identity_process_returns_equivalent_cutset() -> None:
    cs = CutSet([_cut("c0"), _cut("c1"), _cut("c2")])
    op = IdentityOperator(IdentityConfig(), ctx=object())  # type: ignore[arg-type]
    result = op.process(cs)
    assert [c.id for c in result] == ["c0", "c1", "c2"]


def test_identity_config_has_no_required_fields() -> None:
    cfg = IdentityConfig()
    assert cfg is not None
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/operators/test_identity.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/voxkitchen/operators/noop/identity.py`**

```python
"""IdentityOperator: the simplest possible operator (no-op passthrough).

Ships with Plan 2 primarily as a test scaffold. It has no parameters, no
dependencies, and does not touch audio bytes — so integration tests can
verify pipeline orchestration end-to-end without needing ffmpeg, torch,
or any real data.
"""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class IdentityConfig(OperatorConfig):
    """IdentityOperator has no tunable parameters."""


@register_operator
class IdentityOperator(Operator):
    name = "identity"
    config_cls = IdentityConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False  # identity never touches samples

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(list(cuts))
```

- [ ] **Step 4: Update `src/voxkitchen/operators/noop/__init__.py`**

```python
"""No-op operators used as test scaffolding."""

from voxkitchen.operators.noop.identity import IdentityConfig, IdentityOperator

__all__ = ["IdentityConfig", "IdentityOperator"]
```

- [ ] **Step 5: Update `src/voxkitchen/operators/__init__.py` to trigger registration on import**

Replace with:

```python
"""VoxKitchen operators: transformations from CutSet to CutSet."""

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import (
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
    list_operators,
    register_operator,
)

# Register all built-in operators by importing them. Every built-in module
# must be imported here so that ``get_operator(...)`` can find it at runtime.
from voxkitchen.operators.noop import identity as _noop_identity  # noqa: F401

__all__ = [
    "MissingExtrasError",
    "Operator",
    "OperatorConfig",
    "UnknownOperatorError",
    "get_operator",
    "list_operators",
    "register_operator",
]
```

- [ ] **Step 6: Run the test, see it pass**

```bash
pytest tests/unit/operators/test_identity.py -v
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/operators/noop src/voxkitchen/operators/__init__.py tests/unit/operators/test_identity.py
git commit -m "feat(operators): add IdentityOperator as test scaffold"
```

---

## Task 5: PipelineSpec / StageSpec / IngestSpec (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_spec.py`
- Create: `src/voxkitchen/pipeline/spec.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_spec.py`:

```python
"""Unit tests for voxkitchen.pipeline.spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec


def _ingest() -> IngestSpec:
    return IngestSpec(source="manifest", args={"path": "/tmp/in.jsonl.gz"})


def _stages(*names: str) -> list[StageSpec]:
    return [StageSpec(name=n, op="identity", args={}) for n in names]


def test_stage_spec_construction() -> None:
    s = StageSpec(name="vad", op="silero_vad", args={"threshold": 0.5})
    assert s.name == "vad"
    assert s.op == "silero_vad"
    assert s.args == {"threshold": 0.5}


def test_stage_spec_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        StageSpec.model_validate({"name": "x", "op": "y", "args": {}, "extra": "boom"})


def test_ingest_spec_requires_known_source() -> None:
    with pytest.raises(ValidationError):
        IngestSpec.model_validate({"source": "smoke_signal", "args": {}})


def test_pipeline_spec_minimal() -> None:
    spec = PipelineSpec(
        version="0.1",
        name="demo",
        work_dir="/tmp/work",
        ingest=_ingest(),
        stages=_stages("one", "two"),
    )
    assert spec.version == "0.1"
    assert spec.num_gpus == 1
    assert spec.gc_mode == "aggressive"
    assert spec.description == ""
    assert spec.num_cpu_workers is None


def test_pipeline_spec_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        PipelineSpec(
            version="0.1",
            name="demo",
            work_dir="/tmp/work",
            ingest=_ingest(),
            stages=[
                StageSpec(name="dup", op="identity"),
                StageSpec(name="dup", op="identity"),
            ],
        )


def test_pipeline_spec_rejects_invalid_gc_mode() -> None:
    with pytest.raises(ValidationError):
        PipelineSpec.model_validate(
            {
                "version": "0.1",
                "name": "demo",
                "work_dir": "/tmp/work",
                "gc_mode": "paranoid",
                "ingest": {"source": "manifest", "args": {}},
                "stages": [],
            }
        )


def test_pipeline_spec_requires_at_least_one_stage() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        PipelineSpec(
            version="0.1",
            name="demo",
            work_dir="/tmp/work",
            ingest=_ingest(),
            stages=[],
        )


def test_pipeline_spec_forbids_extra_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineSpec.model_validate(
            {
                "version": "0.1",
                "name": "demo",
                "work_dir": "/tmp/work",
                "ingest": {"source": "manifest", "args": {}},
                "stages": [{"name": "s", "op": "identity", "args": {}}],
                "future_field": "boom",
            }
        )
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_spec.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/spec.py`**

```python
"""Pydantic models for pipeline YAML.

These are the canonical in-memory representation of a pipeline. The YAML
loader in ``loader.py`` is responsible for turning raw ``dict`` into these
objects (including string interpolation); the runner consumes them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class StageSpec(BaseModel):
    """One stage in a pipeline — a named invocation of an operator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    op: str
    args: dict[str, Any] = {}


class IngestSpec(BaseModel):
    """How Cuts enter the pipeline before any stage runs."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["dir", "manifest", "recipe"]
    args: dict[str, Any] = {}
    recipe: str | None = None


class PipelineSpec(BaseModel):
    """Top-level pipeline specification."""

    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    description: str = ""
    work_dir: str
    num_gpus: int = 1
    num_cpu_workers: int | None = None
    gc_mode: Literal["aggressive", "keep"] = "aggressive"
    ingest: IngestSpec
    stages: list[StageSpec]

    @field_validator("stages")
    @classmethod
    def _stages_non_empty_and_unique(cls, v: list[StageSpec]) -> list[StageSpec]:
        if len(v) == 0:
            raise ValueError("pipeline must declare at least one stage")
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(
                f"stage names must be unique, duplicates: {sorted(set(duplicates))}"
            )
        return v
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""

from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec

__all__ = ["IngestSpec", "PipelineSpec", "StageSpec"]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_spec.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/spec.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_spec.py
git commit -m "feat(pipeline): add PipelineSpec / StageSpec / IngestSpec models"
```

---

## Task 6: YAML loader with interpolation (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_loader.py`
- Create: `src/voxkitchen/pipeline/loader.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_loader.py`:

```python
"""Unit tests for voxkitchen.pipeline.loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_minimal_pipeline(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest:
  source: manifest
  args:
    path: /tmp/cuts.jsonl.gz
stages:
  - name: one
    op: identity
""",
    )
    spec = load_pipeline_spec(path)
    assert spec.name == "demo"
    assert spec.ingest.source == "manifest"
    assert len(spec.stages) == 1
    assert spec.stages[0].name == "one"


def test_expands_name_and_run_id_in_work_dir(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/${name}-${run_id}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-FIXED")
    assert spec.work_dir == "/tmp/demo-run-FIXED"


def test_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/data/librispeech")
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: ${env:DATA_ROOT}/work
ingest: { source: manifest, args: { path: "${env:DATA_ROOT}/cuts.jsonl.gz" } }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-x")
    assert spec.work_dir == "/data/librispeech/work"
    assert spec.ingest.args["path"] == "/data/librispeech/cuts.jsonl.gz"


def test_unresolved_env_var_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: ${env:NOPE_NEVER_SET}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    with pytest.raises(PipelineLoadError, match="NOPE_NEVER_SET"):
        load_pipeline_spec(path, run_id="run-x")


def test_interpolation_does_not_touch_non_string_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
num_gpus: 4
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity, args: { threshold: 0.5 } }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-x")
    assert spec.num_gpus == 4
    assert spec.stages[0].args["threshold"] == 0.5


def test_invalid_yaml_raises_load_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: '0.1'\nname: [this is: broken")
    with pytest.raises(PipelineLoadError):
        load_pipeline_spec(path, run_id="run-x")


def test_schema_validation_errors_wrapped_in_load_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: smoke_signal, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    with pytest.raises(PipelineLoadError, match="source"):
        load_pipeline_spec(path, run_id="run-x")


def test_load_generates_run_id_when_not_provided(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/${run_id}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path)
    assert spec.work_dir.startswith("/tmp/run-")
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_loader.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/loader.py`**

```python
"""YAML → PipelineSpec loader with string interpolation.

Supported placeholders inside any string value:

- ``${name}`` — the top-level pipeline ``name``
- ``${run_id}`` — the pipeline run id (generated or provided by the caller)
- ``${env:VAR_NAME}`` — value of the environment variable ``VAR_NAME``

Placeholders are resolved by a single recursive pass over the parsed YAML.
Non-string values pass through unchanged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from voxkitchen.pipeline.spec import PipelineSpec
from voxkitchen.utils.run_id import generate_run_id

_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


class PipelineLoadError(ValueError):
    """Raised when a pipeline YAML cannot be parsed, interpolated, or validated."""


def load_pipeline_spec(path: Path, run_id: str | None = None) -> PipelineSpec:
    """Load a pipeline YAML file and return a validated ``PipelineSpec``.

    ``run_id`` is used to expand ``${run_id}`` placeholders. If omitted, a
    fresh id is generated via ``generate_run_id()``.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineLoadError(f"cannot read {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise PipelineLoadError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineLoadError(
            f"pipeline YAML must be a mapping at the top level, got {type(raw).__name__}"
        )

    effective_run_id = run_id if run_id is not None else generate_run_id()
    pipeline_name = raw.get("name", "")

    interpolated = _interpolate(
        raw, name=str(pipeline_name), run_id=effective_run_id
    )

    try:
        return PipelineSpec.model_validate(interpolated)
    except ValidationError as exc:
        raise PipelineLoadError(f"pipeline validation failed for {path}:\n{exc}") from exc


def _interpolate(obj: Any, *, name: str, run_id: str) -> Any:
    """Recursively replace ``${...}`` placeholders in string values."""
    if isinstance(obj, str):
        return _interpolate_string(obj, name=name, run_id=run_id)
    if isinstance(obj, dict):
        return {k: _interpolate(v, name=name, run_id=run_id) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, name=name, run_id=run_id) for v in obj]
    return obj


def _interpolate_string(s: str, *, name: str, run_id: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "name":
            return name
        if key == "run_id":
            return run_id
        if key.startswith("env:"):
            env_var = key[4:]
            value = os.environ.get(env_var)
            if value is None:
                raise PipelineLoadError(
                    f"environment variable {env_var!r} referenced by ${{env:{env_var}}} "
                    f"is not set"
                )
            return value
        raise PipelineLoadError(f"unknown placeholder: ${{{key}}}")

    return _INTERP_RE.sub(replace, s)
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec

__all__ = [
    "IngestSpec",
    "PipelineLoadError",
    "PipelineSpec",
    "StageSpec",
    "load_pipeline_spec",
]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_loader.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/loader.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_loader.py
git commit -m "feat(pipeline): add YAML loader with name/run_id/env interpolation"
```

---

## Task 7: RunContext (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_context.py`
- Create: `src/voxkitchen/pipeline/context.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_context.py`:

```python
"""Unit tests for voxkitchen.pipeline.context.RunContext."""

from __future__ import annotations

import pickle
from pathlib import Path

from voxkitchen.pipeline.context import RunContext


def _ctx(work_dir: Path) -> RunContext:
    return RunContext(
        work_dir=work_dir,
        pipeline_run_id="run-abcd",
        stage_index=2,
        stage_name="vad",
        num_gpus=4,
        num_cpu_workers=8,
        gc_mode="aggressive",
        device="cpu",
    )


def test_run_context_stage_dir_uses_zero_padded_index(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.stage_dir == tmp_path / "02_vad"


def test_run_context_with_stage_returns_new_instance(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    new_ctx = ctx.with_stage(stage_index=5, stage_name="asr")
    assert new_ctx.stage_index == 5
    assert new_ctx.stage_name == "asr"
    # original unchanged
    assert ctx.stage_index == 2
    assert ctx.stage_name == "vad"
    # other fields inherited
    assert new_ctx.pipeline_run_id == ctx.pipeline_run_id
    assert new_ctx.num_gpus == ctx.num_gpus


def test_run_context_is_picklable(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    blob = pickle.dumps(ctx)
    restored = pickle.loads(blob)
    assert restored == ctx


def test_ingest_dir_is_stage_zero(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    ingest_ctx = ctx.with_stage(stage_index=0, stage_name="ingest")
    assert ingest_ctx.stage_dir == tmp_path / "00_ingest"
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_context.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/context.py`**

```python
"""Runtime context passed to operators and executors.

RunContext is a small dataclass containing everything an operator might need
at runtime (paths, identifiers, device assignment) without pulling in any
unpicklable objects like file handles or logger instances. This is crucial
for the multiprocessing-based executors — ``RunContext`` crosses process
boundaries via pickle, so it must stay simple.

Logging is handled through ``logging.getLogger(...)`` inside each worker;
loggers are looked up by name, not passed around.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunContext:
    """Pickle-friendly runtime context for a single stage of a pipeline run."""

    work_dir: Path
    pipeline_run_id: str
    stage_index: int
    stage_name: str
    num_gpus: int
    num_cpu_workers: int
    gc_mode: Literal["aggressive", "keep"]
    device: str  # "cpu" or "cuda:0" from the worker's view

    @property
    def stage_dir(self) -> Path:
        """Directory for this stage's outputs: ``work_dir/NN_<name>/``."""
        return self.work_dir / f"{self.stage_index:02d}_{self.stage_name}"

    def with_stage(self, *, stage_index: int, stage_name: str) -> "RunContext":
        """Return a copy of this context advanced to a new stage."""
        return replace(self, stage_index=stage_index, stage_name=stage_name)
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""

from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec

__all__ = [
    "IngestSpec",
    "PipelineLoadError",
    "PipelineSpec",
    "RunContext",
    "StageSpec",
    "load_pipeline_spec",
]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_context.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/context.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_context.py
git commit -m "feat(pipeline): add RunContext dataclass (pickle-friendly)"
```

---

## Task 8: Checkpoint / resume helpers (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_checkpoint.py`
- Create: `src/voxkitchen/pipeline/checkpoint.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_checkpoint.py`:

```python
"""Unit tests for voxkitchen.pipeline.checkpoint."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)


def test_stage_dir_name_is_zero_padded() -> None:
    assert stage_dir_name(0, "ingest") == "00_ingest"
    assert stage_dir_name(9, "vad") == "09_vad"
    assert stage_dir_name(10, "asr") == "10_asr"


def test_write_success_marker_creates_empty_file(tmp_path: Path) -> None:
    stage_dir = tmp_path / "00_ingest"
    stage_dir.mkdir()
    write_success_marker(stage_dir)
    marker = stage_dir / "_SUCCESS"
    assert marker.exists()
    assert marker.read_bytes() == b""


def test_is_stage_complete_requires_both_manifest_and_marker(tmp_path: Path) -> None:
    stage_dir = tmp_path / "01_vad"
    stage_dir.mkdir()

    assert not is_stage_complete(stage_dir)  # neither file
    (stage_dir / "cuts.jsonl.gz").write_bytes(b"x")
    assert not is_stage_complete(stage_dir)  # manifest only
    write_success_marker(stage_dir)
    assert is_stage_complete(stage_dir)  # both present


def test_is_stage_complete_false_if_directory_missing(tmp_path: Path) -> None:
    assert not is_stage_complete(tmp_path / "does-not-exist")


def test_find_last_completed_stage_returns_none_when_nothing_complete(tmp_path: Path) -> None:
    result = find_last_completed_stage(tmp_path, ["ingest", "vad", "asr"])
    assert result is None


def test_find_last_completed_stage_returns_highest_complete_index(tmp_path: Path) -> None:
    for i, name in enumerate(["ingest", "vad", "asr"]):
        d = tmp_path / stage_dir_name(i, name)
        d.mkdir()
    # Complete stages 0 and 1 but not 2
    (tmp_path / "00_ingest" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "00_ingest")
    (tmp_path / "01_vad" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "01_vad")

    assert find_last_completed_stage(tmp_path, ["ingest", "vad", "asr"]) == 1


def test_find_last_completed_stage_stops_at_first_incomplete(tmp_path: Path) -> None:
    """If stage 1 is incomplete but stage 2 is complete, treat stage 0 as the resume point."""
    for i, name in enumerate(["a", "b", "c"]):
        d = tmp_path / stage_dir_name(i, name)
        d.mkdir()
    (tmp_path / "00_a" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "00_a")
    # stage 1 incomplete (manifest without success)
    (tmp_path / "01_b" / "cuts.jsonl.gz").write_bytes(b"x")
    # stage 2 "complete" (but shouldn't count due to gap)
    (tmp_path / "02_c" / "cuts.jsonl.gz").write_bytes(b"x")
    write_success_marker(tmp_path / "02_c")

    assert find_last_completed_stage(tmp_path, ["a", "b", "c"]) == 0
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_checkpoint.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/checkpoint.py`**

```python
"""Resume / checkpoint helpers.

A stage is considered complete iff both ``cuts.jsonl.gz`` and ``_SUCCESS``
exist in its directory. The ``_SUCCESS`` marker is written by the runner
only after the manifest has been fully flushed to disk — a crash during
write leaves the manifest present but the marker absent, and the runner
will re-run the stage on the next attempt.
"""

from __future__ import annotations

from pathlib import Path


def stage_dir_name(index: int, name: str) -> str:
    """Return the canonical on-disk directory name: ``NN_<name>``."""
    return f"{index:02d}_{name}"


def write_success_marker(stage_dir: Path) -> None:
    """Create the empty ``_SUCCESS`` marker inside a stage directory."""
    (stage_dir / "_SUCCESS").touch()


def is_stage_complete(stage_dir: Path) -> bool:
    """Return True iff both the manifest and the success marker exist."""
    if not stage_dir.is_dir():
        return False
    return (stage_dir / "cuts.jsonl.gz").exists() and (stage_dir / "_SUCCESS").exists()


def find_last_completed_stage(work_dir: Path, stage_names: list[str]) -> int | None:
    """Return the index of the last contiguously-completed stage, or None.

    Scans stages in order and stops at the first gap — resume only restarts
    from the last known-good point, never "skips" an incomplete stage.
    """
    last_complete: int | None = None
    for i, name in enumerate(stage_names):
        stage_dir = work_dir / stage_dir_name(i, name)
        if is_stage_complete(stage_dir):
            last_complete = i
        else:
            break
    return last_complete
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""

from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec

__all__ = [
    "IngestSpec",
    "PipelineLoadError",
    "PipelineSpec",
    "RunContext",
    "StageSpec",
    "find_last_completed_stage",
    "is_stage_complete",
    "load_pipeline_spec",
    "stage_dir_name",
    "write_success_marker",
]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_checkpoint.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/checkpoint.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_checkpoint.py
git commit -m "feat(pipeline): add checkpoint/_SUCCESS resume helpers"
```

---

## Task 9: Garbage collection plan + execution (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_gc.py`
- Create: `src/voxkitchen/pipeline/gc.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_gc.py`:

```python
"""Unit tests for voxkitchen.pipeline.gc."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import _REGISTRY, register_operator
from voxkitchen.pipeline.checkpoint import stage_dir_name
from voxkitchen.pipeline.gc import (
    GcPlan,
    compute_gc_plan,
    empty_trash,
    run_gc,
)
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec
from voxkitchen.schema.cutset import CutSet


@pytest.fixture(autouse=True)
def _registry_snapshot() -> None:
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _register_ops() -> None:
    class _Config(OperatorConfig):
        pass

    class MaterializeOp(Operator):
        name = "materialize"
        config_cls = _Config
        produces_audio = True
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class AudioReaderOp(Operator):
        name = "audio_reader"
        config_cls = _Config
        produces_audio = False
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class MetricOnlyOp(Operator):
        name = "metric_only"
        config_cls = _Config
        produces_audio = False
        reads_audio_bytes = False

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    class PackOp(Operator):
        name = "pack"
        config_cls = _Config
        produces_audio = True
        reads_audio_bytes = True

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    register_operator(MaterializeOp)
    register_operator(AudioReaderOp)
    register_operator(MetricOnlyOp)
    register_operator(PackOp)


def _spec(*stage_ops: str) -> PipelineSpec:
    return PipelineSpec(
        version="0.1",
        name="demo",
        work_dir="/tmp/work",
        ingest=IngestSpec(source="manifest", args={}),
        stages=[StageSpec(name=f"s{i}", op=op) for i, op in enumerate(stage_ops)],
    )


def test_gc_plan_empty_for_pipeline_without_materializers() -> None:
    _register_ops()
    spec = _spec("audio_reader", "metric_only")
    plan = compute_gc_plan(spec)
    assert plan.last_consumer == {}


def test_gc_plan_maps_materializer_to_last_consumer() -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader", "metric_only")
    plan = compute_gc_plan(spec)
    # materialize @ index 0 is consumed by audio_reader @ index 1
    assert plan.last_consumer == {0: 1}


def test_gc_plan_for_consecutive_materializers() -> None:
    _register_ops()
    spec = _spec("materialize", "materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    # s0 is consumed by s1 (materialize reads audio); s1 is consumed by s2
    assert plan.last_consumer == {0: 1, 1: 2}


def test_gc_plan_final_pack_stage_is_excluded_from_gc() -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader", "pack")
    plan = compute_gc_plan(spec)
    # pack is the last stage and its output is the user-facing artifact, never GC'd
    assert 2 not in plan.last_consumer


def test_gc_plan_materializer_with_no_downstream_consumer_has_no_entry() -> None:
    _register_ops()
    # materialize is followed only by a metric-only stage, so nothing "consumes" it
    spec = _spec("materialize", "metric_only")
    plan = compute_gc_plan(spec)
    assert plan.last_consumer == {}


def test_run_gc_moves_derived_to_trash(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    # Lay out simulated work_dir
    s0 = tmp_path / stage_dir_name(0, "s0")
    s1 = tmp_path / stage_dir_name(1, "s1")
    (s0 / "derived").mkdir(parents=True)
    (s0 / "derived" / "out.wav").write_bytes(b"fake")
    s1.mkdir(parents=True)

    # After s1 (index 1) completes, s0's derived/ is eligible for GC
    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=1,
        gc_mode="aggressive",
        stage_names=stage_names,
    )

    trashed = tmp_path / "derived_trash" / stage_dir_name(0, "s0") / "derived"
    assert trashed.exists()
    assert (trashed / "out.wav").exists()
    assert not (s0 / "derived").exists()


def test_run_gc_is_noop_in_keep_mode(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    s0 = tmp_path / stage_dir_name(0, "s0")
    (s0 / "derived").mkdir(parents=True)
    (s0 / "derived" / "out.wav").write_bytes(b"fake")

    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=1,
        gc_mode="keep",
        stage_names=stage_names,
    )
    assert (s0 / "derived" / "out.wav").exists()
    assert not (tmp_path / "derived_trash").exists()


def test_run_gc_does_nothing_if_stage_not_producer(tmp_path: Path) -> None:
    _register_ops()
    spec = _spec("audio_reader", "materialize", "audio_reader")
    plan = compute_gc_plan(spec)
    stage_names = [s.name for s in spec.stages]

    s1 = tmp_path / stage_dir_name(1, "s1")
    (s1 / "derived").mkdir(parents=True)
    (s1 / "derived" / "out.wav").write_bytes(b"fake")

    # After s0 completes, nothing should be GC'd (s0 doesn't produce audio)
    run_gc(
        plan,
        work_dir=tmp_path,
        just_completed_idx=0,
        gc_mode="aggressive",
        stage_names=stage_names,
    )
    assert (s1 / "derived" / "out.wav").exists()


def test_empty_trash_removes_trash_dir(tmp_path: Path) -> None:
    trash = tmp_path / "derived_trash"
    (trash / "sub").mkdir(parents=True)
    (trash / "sub" / "file").write_bytes(b"x")
    empty_trash(tmp_path)
    assert not trash.exists()


def test_empty_trash_is_noop_when_trash_missing(tmp_path: Path) -> None:
    empty_trash(tmp_path)  # must not raise
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_gc.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/gc.py`**

```python
"""Aggressive garbage collection for materialized audio in ``derived/`` dirs.

The plan is computed once at pipeline startup by scanning the stage list for
operators with ``produces_audio = True`` and finding, for each such producer,
the last downstream stage whose operator has ``reads_audio_bytes = True``.
That downstream stage is the ``last_consumer`` — once it completes, the
producer's derived files are no longer needed and can be moved to
``derived_trash/``.

``derived_trash/`` is emptied only after the entire pipeline completes
successfully. A crash during the run leaves trash in place for diagnostics.

The final stage is never GC'd — its output is the user-facing artifact.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.checkpoint import stage_dir_name
from voxkitchen.pipeline.spec import PipelineSpec


@dataclass
class GcPlan:
    """Static analysis result: producer stage index → last consumer stage index."""

    last_consumer: dict[int, int] = field(default_factory=dict)


def compute_gc_plan(spec: PipelineSpec) -> GcPlan:
    """Scan a PipelineSpec and produce a GcPlan.

    For each stage whose operator ``produces_audio`` (and is not the final
    stage), find the last downstream stage whose operator ``reads_audio_bytes``.
    Record the pair as ``(producer_idx, last_consumer_idx)`` in the plan.
    """
    last_consumer: dict[int, int] = {}
    num_stages = len(spec.stages)

    for i, stage in enumerate(spec.stages):
        op_cls = get_operator(stage.op)
        if not op_cls.produces_audio:
            continue
        if i == num_stages - 1:
            # Final stage is the artifact; never GC'd
            continue

        # Find the last downstream consumer that reads audio bytes
        consumer_idx: int | None = None
        for j in range(i + 1, num_stages):
            downstream_op = get_operator(spec.stages[j].op)
            if downstream_op.reads_audio_bytes:
                consumer_idx = j
        if consumer_idx is not None:
            last_consumer[i] = consumer_idx

    return GcPlan(last_consumer=last_consumer)


def run_gc(
    plan: GcPlan,
    *,
    work_dir: Path,
    just_completed_idx: int,
    gc_mode: Literal["aggressive", "keep"],
    stage_names: list[str] | None = None,
) -> None:
    """Move any now-unneeded ``derived/`` directories to ``derived_trash/``.

    ``stage_names``, if provided, is used to resolve producer indices to
    on-disk directory names. When omitted, callers are expected to pass it
    because directory names embed the stage name, not just the index.
    """
    if gc_mode == "keep":
        return
    if stage_names is None:
        stage_names = []

    trash_root = work_dir / "derived_trash"

    for producer_idx, consumer_idx in plan.last_consumer.items():
        if consumer_idx != just_completed_idx:
            continue
        if producer_idx >= len(stage_names):
            continue
        producer_name = stage_names[producer_idx]
        producer_dir = work_dir / stage_dir_name(producer_idx, producer_name)
        derived = producer_dir / "derived"
        if not derived.exists():
            continue
        target = trash_root / stage_dir_name(producer_idx, producer_name) / "derived"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(derived), str(target))


def empty_trash(work_dir: Path) -> None:
    """Permanently delete the ``derived_trash/`` directory. Safe to call twice."""
    trash = work_dir / "derived_trash"
    if trash.exists():
        shutil.rmtree(trash)
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

```python
"""VoxKitchen pipeline engine: spec, loader, runner, executors."""

from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.gc import GcPlan, compute_gc_plan, empty_trash, run_gc
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec

__all__ = [
    "GcPlan",
    "IngestSpec",
    "PipelineLoadError",
    "PipelineSpec",
    "RunContext",
    "StageSpec",
    "compute_gc_plan",
    "empty_trash",
    "find_last_completed_stage",
    "is_stage_complete",
    "load_pipeline_spec",
    "run_gc",
    "stage_dir_name",
    "write_success_marker",
]
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_gc.py -v
```

Expected: 10 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/gc.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_gc.py
git commit -m "feat(pipeline): add GC plan and trash-based derived file cleanup"
```

---

## Task 10: CpuPoolExecutor (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_executor_cpu.py`
- Create: `src/voxkitchen/pipeline/executor.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

**Note:** This task introduces the `Executor` Protocol and the CPU implementation. `GpuPoolExecutor` lands in Task 11 in the same file.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_executor_cpu.py`:

```python
"""Unit tests for voxkitchen.pipeline.executor.CpuPoolExecutor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.noop.identity import IdentityConfig, IdentityOperator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import CpuPoolExecutor
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-a1b2c3",
        stage_index=1,
        stage_name="identity",
        num_gpus=0,
        num_cpu_workers=2,
        gc_mode="aggressive",
        device="cpu",
    )


def test_cpu_pool_executor_preserves_all_cuts(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(10)])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=2)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert sorted(c.id for c in result) == sorted(c.id for c in cs)


def test_cpu_pool_executor_with_single_worker(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(3)])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=1)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert [c.id for c in result] == [c.id for c in cs]


def test_cpu_pool_executor_handles_empty_cutset(tmp_path: Path) -> None:
    cs = CutSet([])
    ctx = _ctx(tmp_path)
    executor = CpuPoolExecutor(num_workers=2)
    result = executor.run(IdentityOperator, IdentityConfig(), cs, ctx)
    assert len(result) == 0
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_executor_cpu.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/executor.py`** (CPU portion; GPU added in Task 11)

```python
"""Executors run a single Operator over a CutSet.

Two implementations ship with Plan 2:

- ``CpuPoolExecutor`` — splits the CutSet into N shards and runs an Operator
  worker in a ``multiprocessing.Pool`` (spawn context) over each shard.
- ``GpuPoolExecutor`` — added in Task 11. Spawns N subprocesses each pinned to
  one GPU via ``CUDA_VISIBLE_DEVICES`` before importing torch.

Both executors share the same Protocol. The runner picks between them based
on ``Operator.device``.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Protocol, cast

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cutset import CutSet


class Executor(Protocol):
    """An Executor knows how to run one Operator over a CutSet."""

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet: ...


def _cpu_worker(
    op_cls: type[Operator],
    config_json: str,
    ctx: RunContext,
    cuts_list: list,
) -> list:
    """Instantiate op, call setup/process/teardown, return processed cuts.

    Config is passed as JSON (not the Pydantic instance) because some
    Pydantic models pickle awkwardly across spawn boundaries; JSON is safe.
    The operator reconstructs its config from JSON inside the worker.
    Cuts are passed as a list of Cut instances (Pydantic v2 pickles cleanly).
    """
    config = op_cls.config_cls.model_validate_json(config_json)
    op = op_cls(config, ctx)
    op.setup()
    try:
        input_cuts = CutSet(cuts_list)
        output_cuts = op.process(input_cuts)
        return list(output_cuts)
    finally:
        op.teardown()


class CpuPoolExecutor:
    """Shard a CutSet across a multiprocessing.Pool of CPU workers."""

    def __init__(self, num_workers: int) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be ≥ 1, got {num_workers}")
        self.num_workers = num_workers

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet:
        if len(cuts) == 0:
            return CutSet([])

        effective_workers = min(self.num_workers, len(cuts))
        shards = cuts.split(effective_workers)
        config_json = config.model_dump_json()

        if effective_workers == 1:
            # Skip the pool entirely — simpler + faster for single worker
            result = _cpu_worker(op_cls, config_json, ctx, list(shards[0]))
            return CutSet(result)

        tasks = [(op_cls, config_json, ctx, list(shard)) for shard in shards]

        ctx_mp = mp.get_context("spawn")
        with ctx_mp.Pool(effective_workers) as pool:
            results = pool.starmap(_cpu_worker, tasks)

        merged: list = []
        for shard_result in results:
            merged.extend(shard_result)
        return CutSet(merged)
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

Add `CpuPoolExecutor` and `Executor` to the existing `__init__.py`:

```python
from voxkitchen.pipeline.executor import CpuPoolExecutor, Executor
```

and add to `__all__`.

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_executor_cpu.py -v
```

Expected: 3 passed. (If you see pickling errors, verify that `IdentityOperator` is importable from a fresh process — since it's registered in `voxkitchen/operators/__init__.py` via an import, this should work.)

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/executor.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_executor_cpu.py
git commit -m "feat(pipeline): add CpuPoolExecutor with spawn multiprocessing"
```

---

## Task 11: GpuPoolExecutor (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_executor_gpu.py`
- Modify: `src/voxkitchen/pipeline/executor.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_executor_gpu.py`:

```python
"""Unit tests for voxkitchen.pipeline.executor.GpuPoolExecutor.

The tests don't require a real GPU. ``IdentityOperator`` never touches torch,
so the worker process runs happily on CPU-only CI runners. What we verify is
the orchestration: shards are processed by the right number of workers, all
cuts are recovered in the output, and CUDA_VISIBLE_DEVICES is set per worker.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import GpuPoolExecutor
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance


class _CudaSentinelConfig(OperatorConfig):
    pass


@register_operator
class _CudaSentinelOperator(Operator):
    """Stamps each Cut's custom dict with the CUDA_VISIBLE_DEVICES it saw."""

    name = "_test_cuda_sentinel"
    config_cls = _CudaSentinelConfig
    device = "gpu"

    def process(self, cuts: CutSet) -> CutSet:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET")
        out = []
        for c in cuts:
            out.append(c.model_copy(update={"custom": {"cvd": cvd}}))
        return CutSet(out)


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-a1b2c3",
        stage_index=1,
        stage_name="gpu_stage",
        num_gpus=2,
        num_cpu_workers=0,
        gc_mode="aggressive",
        device="cuda:0",
    )


def test_gpu_pool_executor_recovers_all_cuts(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(6)])
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    assert sorted(c.id for c in result) == sorted(c.id for c in cs)


def test_gpu_pool_executor_sets_cuda_visible_devices_per_worker(tmp_path: Path) -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(4)])
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    # Each Cut should carry the CUDA_VISIBLE_DEVICES its worker saw.
    seen = {c.custom["cvd"] for c in result}
    # Should contain "0" and "1" at least (shards may be uneven but both GPUs used
    # since we have 4 cuts and 2 workers)
    assert seen == {"0", "1"}


def test_gpu_pool_executor_empty_cutset(tmp_path: Path) -> None:
    executor = GpuPoolExecutor(num_gpus=2)
    result = executor.run(
        _CudaSentinelOperator, _CudaSentinelConfig(), CutSet([]), _ctx(tmp_path)
    )
    assert len(result) == 0


def test_gpu_pool_executor_uses_single_gpu_when_fewer_cuts(tmp_path: Path) -> None:
    """With 1 cut and num_gpus=4, only 1 worker should be spawned."""
    cs = CutSet([_cut("only")])
    executor = GpuPoolExecutor(num_gpus=4)
    result = executor.run(_CudaSentinelOperator, _CudaSentinelConfig(), cs, _ctx(tmp_path))
    assert len(result) == 1
    only_cut = next(iter(result))
    assert only_cut.custom["cvd"] == "0"
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_executor_gpu.py -v
```

Expected: `ImportError: cannot import name 'GpuPoolExecutor'`.

- [ ] **Step 3: Append `GpuPoolExecutor` to `src/voxkitchen/pipeline/executor.py`**

Add this code to the end of `executor.py`:

```python
def _gpu_worker(
    gpu_id: int,
    op_cls: type[Operator],
    config_json: str,
    ctx: RunContext,
    cuts_list: list,
) -> list:
    """GPU worker entry point.

    Sets ``CUDA_VISIBLE_DEVICES`` to its assigned id BEFORE torch has any
    chance to be imported. The operator's ``setup()`` (called after this)
    can safely ``import torch`` and see only the intended GPU as ``cuda:0``.
    """
    import os as _os

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Rewrite ctx.device to reflect worker-local view
    from dataclasses import replace as _replace

    worker_ctx = _replace(ctx, device="cuda:0")

    config = op_cls.config_cls.model_validate_json(config_json)
    op = op_cls(config, worker_ctx)
    op.setup()
    try:
        input_cuts = CutSet(cuts_list)
        output_cuts = op.process(input_cuts)
        return list(output_cuts)
    finally:
        op.teardown()


class GpuPoolExecutor:
    """Shard a CutSet across N GPU-pinned subprocesses.

    Each worker is a fresh ``spawn`` process with ``CUDA_VISIBLE_DEVICES=i``
    set before any torch import. Operators that need GPUs use ``cuda:0`` in
    their ``setup()`` — that maps to the correct physical GPU via the env var.
    """

    def __init__(self, num_gpus: int) -> None:
        if num_gpus < 1:
            raise ValueError(f"num_gpus must be ≥ 1, got {num_gpus}")
        self.num_gpus = num_gpus

    def run(
        self,
        op_cls: type[Operator],
        config: OperatorConfig,
        cuts: CutSet,
        ctx: RunContext,
    ) -> CutSet:
        if len(cuts) == 0:
            return CutSet([])

        effective_workers = min(self.num_gpus, len(cuts))
        shards = cuts.split(effective_workers)
        config_json = config.model_dump_json()

        tasks = [
            (gpu_id, op_cls, config_json, ctx, list(shard))
            for gpu_id, shard in enumerate(shards)
        ]

        ctx_mp = mp.get_context("spawn")
        with ctx_mp.Pool(effective_workers) as pool:
            results = pool.starmap(_gpu_worker, tasks)

        merged: list = []
        for shard_result in results:
            merged.extend(shard_result)
        return CutSet(merged)
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`** to also export `GpuPoolExecutor`.

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_executor_gpu.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/executor.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_executor_gpu.py
git commit -m "feat(pipeline): add GpuPoolExecutor with CUDA_VISIBLE_DEVICES pinning"
```

---

## Task 12: Manifest ingest source (TDD)

**Files:**
- Create: `tests/unit/ingest/test_manifest_import.py`
- Create: `src/voxkitchen/ingest/base.py`
- Create: `src/voxkitchen/ingest/manifest_import.py`
- Modify: `src/voxkitchen/ingest/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ingest/test_manifest_import.py`:

```python
"""Unit tests for voxkitchen.ingest.manifest_import.ManifestIngestSource."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.ingest.manifest_import import (
    ManifestIngestConfig,
    ManifestIngestSource,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="fixture",
        ),
    )


def _write_manifest(path: Path, cs: CutSet) -> None:
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture",
        stage_name="00_ingest",
    )
    cs.to_jsonl_gz(path, header)


def _ctx(work_dir: Path) -> RunContext:
    return RunContext(
        work_dir=work_dir,
        pipeline_run_id="run-a1b2c3",
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def test_manifest_ingest_reads_all_cuts(tmp_path: Path) -> None:
    src_path = tmp_path / "input.jsonl.gz"
    cs = CutSet([_cut(f"c{i}") for i in range(3)])
    _write_manifest(src_path, cs)

    ingest = ManifestIngestSource(
        ManifestIngestConfig(path=str(src_path)), ctx=_ctx(tmp_path)
    )
    result = ingest.run()
    assert sorted(c.id for c in result) == ["c0", "c1", "c2"]


def test_manifest_ingest_rejects_missing_file(tmp_path: Path) -> None:
    ingest = ManifestIngestSource(
        ManifestIngestConfig(path=str(tmp_path / "nope.jsonl.gz")), ctx=_ctx(tmp_path)
    )
    with pytest.raises(FileNotFoundError):
        ingest.run()


def test_manifest_ingest_config_requires_path() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ManifestIngestConfig.model_validate({})
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/ingest/test_manifest_import.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/ingest/base.py`**

```python
"""IngestSource ABC: the common interface for all v0.1 ingest paths.

An IngestSource is responsible for producing the **initial** CutSet that
flows into a pipeline's first stage. Plan 2 ships only ``ManifestIngestSource``;
Plan 3 adds ``DirScanIngestSource`` and the recipe framework.

Unlike Operators, IngestSources take their config at construction time and
expose a simple ``run() -> CutSet`` method. They do not run inside executor
workers — ingest always runs in the main runner process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class IngestConfig(BaseModel):
    """Base class for ingest source parameter models."""

    model_config = ConfigDict(extra="forbid")


class IngestSource(ABC):
    """Base class for all ingest sources."""

    name: str = ""  # overridden by subclasses
    config_cls: type[IngestConfig] = IngestConfig

    def __init__(self, config: IngestConfig, ctx: "RunContext") -> None:
        self.config = config
        self.ctx = ctx

    @abstractmethod
    def run(self) -> CutSet:
        """Produce the initial CutSet for a pipeline."""
```

- [ ] **Step 4: Write `src/voxkitchen/ingest/manifest_import.py`**

```python
"""ManifestIngestSource: read a pre-built ``cuts.jsonl.gz`` as the pipeline input.

This is the simplest possible ingest path. It serves three purposes:

1. A real user-facing option for people who already have a manifest (e.g.,
   from a previous pipeline run or from a third-party tool).
2. The scaffolding for integration tests that need deterministic input.
3. A reference implementation showing how IngestSource subclasses look.
"""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.schema.cutset import CutSet


class ManifestIngestConfig(IngestConfig):
    """Parameters for ``ManifestIngestSource``."""

    path: str  # required: where to read the existing cuts.jsonl.gz


class ManifestIngestSource(IngestSource):
    name = "manifest"
    config_cls = ManifestIngestConfig

    def run(self) -> CutSet:
        assert isinstance(self.config, ManifestIngestConfig)
        path = Path(self.config.path)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        return CutSet.from_jsonl_gz(path)
```

- [ ] **Step 5: Update `src/voxkitchen/ingest/__init__.py`**

```python
"""VoxKitchen ingest sources: how Cuts enter a pipeline."""

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.ingest.manifest_import import (
    ManifestIngestConfig,
    ManifestIngestSource,
)

# Registry of ingest sources keyed by the IngestSpec.source literal
_INGEST_SOURCES: dict[str, type[IngestSource]] = {
    "manifest": ManifestIngestSource,
}


def get_ingest_source(name: str) -> type[IngestSource]:
    if name not in _INGEST_SOURCES:
        raise KeyError(
            f"ingest source {name!r} not available in this build. "
            f"Plan 2 ships: {sorted(_INGEST_SOURCES.keys())}"
        )
    return _INGEST_SOURCES[name]


__all__ = [
    "IngestConfig",
    "IngestSource",
    "ManifestIngestConfig",
    "ManifestIngestSource",
    "get_ingest_source",
]
```

- [ ] **Step 6: Run the test, see it pass**

```bash
pytest tests/unit/ingest/test_manifest_import.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/ingest tests/unit/ingest/test_manifest_import.py
git commit -m "feat(ingest): add IngestSource ABC and ManifestIngestSource"
```

---

## Task 13: Runner (TDD)

**Files:**
- Create: `tests/unit/pipeline/test_runner.py`
- Create: `src/voxkitchen/pipeline/runner.py`
- Modify: `src/voxkitchen/pipeline/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/pipeline/test_runner.py`:

```python
"""Unit tests for voxkitchen.pipeline.runner.run_pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord, read_cuts
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="fixture",
        ),
    )


def _write_input_manifest(path: Path, n: int = 4) -> None:
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture",
        stage_name="00_ingest",
    )
    CutSet([_cut(f"c{i}") for i in range(n)]).to_jsonl_gz(path, header)


def _write_pipeline_yaml(
    path: Path, work_dir: Path, input_manifest: Path, num_stages: int = 2
) -> None:
    stages = "\n".join(
        f"  - {{ name: s{i}, op: identity }}" for i in range(num_stages)
    )
    path.write_text(
        f"""
version: "0.1"
name: runner-test
work_dir: {work_dir}
num_gpus: 1
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
{stages}
""",
        encoding="utf-8",
    )


def test_runner_completes_simple_pipeline(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=2)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    # Both stages should have written _SUCCESS + cuts.jsonl.gz
    assert (work_dir / "00_s0" / "_SUCCESS").exists()
    assert (work_dir / "00_s0" / "cuts.jsonl.gz").exists()
    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert (work_dir / "01_s1" / "cuts.jsonl.gz").exists()


def test_runner_preserves_cut_count_through_stages(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest, n=5)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    final_cuts = list(read_cuts(work_dir / "02_s2" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 5


def test_runner_writes_run_yaml_snapshot(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    run_snapshot = work_dir / "run.yaml"
    assert run_snapshot.exists()
    text = run_snapshot.read_text()
    assert "runner-test" in text
    assert "run-fixed" in text


def test_runner_resume_skips_completed_stages(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest, n=3)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    # First run: complete all stages
    run_pipeline(spec)

    # Delete stage 2's output and re-run — stages 0 and 1 should be skipped
    (work_dir / "02_s2" / "_SUCCESS").unlink()
    (work_dir / "02_s2" / "cuts.jsonl.gz").unlink()

    # Touch stage 1's cuts to ensure it's NOT re-read (we want to verify skip)
    stage1_marker = work_dir / "01_s1" / "cuts.jsonl.gz"
    original_mtime = stage1_marker.stat().st_mtime

    run_pipeline(spec)

    # Stage 1 manifest must not have been rewritten
    assert stage1_marker.stat().st_mtime == original_mtime
    # Stage 2 must now be complete
    assert (work_dir / "02_s2" / "_SUCCESS").exists()


def test_runner_stops_at_stage(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec, stop_at="s1")

    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert not (work_dir / "02_s2").exists()


def test_runner_raises_stage_failed_when_operator_missing(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        f"""
version: "0.1"
name: bad-op
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
  - {{ name: s0, op: nonexistent_operator }}
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")

    with pytest.raises(StageFailedError):
        run_pipeline(spec)
```

- [ ] **Step 2: Run the test, see it fail**

```bash
pytest tests/unit/pipeline/test_runner.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/pipeline/runner.py`**

```python
"""The pipeline runner: orchestrates stages end-to-end.

The runner is the only module that knows about all the other pipeline
pieces at once. It:

1. Resolves the ingest source and produces the initial CutSet.
2. Computes the GC plan.
3. Detects resume point (highest contiguously-completed stage).
4. For each remaining stage:
   a. Instantiates the operator.
   b. Picks an executor based on ``operator.device``.
   c. Runs the operator over the CutSet.
   d. Writes output manifest + _SUCCESS marker.
   e. Runs GC for anything the just-finished stage unblocks.
5. Empties trash on success.

All state is on disk under ``work_dir`` so any crash leaves a resumable tree.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

from voxkitchen.ingest import get_ingest_source
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.checkpoint import (
    find_last_completed_stage,
    is_stage_complete,
    stage_dir_name,
    write_success_marker,
)
from voxkitchen.pipeline.context import RunContext
from voxkitchen.pipeline.executor import (
    CpuPoolExecutor,
    Executor,
    GpuPoolExecutor,
)
from voxkitchen.pipeline.gc import compute_gc_plan, empty_trash, run_gc
from voxkitchen.pipeline.spec import PipelineSpec
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord

logger = logging.getLogger(__name__)


class StageFailedError(RuntimeError):
    """Raised when a stage cannot be completed."""

    def __init__(self, stage_name: str, cause: BaseException) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"stage {stage_name!r} failed: {cause}")


def _make_executor(device: str, ctx: RunContext) -> Executor:
    if device == "gpu":
        return GpuPoolExecutor(num_gpus=max(1, ctx.num_gpus))
    return CpuPoolExecutor(num_workers=max(1, ctx.num_cpu_workers))


def _write_run_snapshot(work_dir: Path, spec: PipelineSpec, run_id: str) -> None:
    snapshot = {
        "__voxkitchen_snapshot__": True,
        "run_id": run_id,
        "spec": spec.model_dump(mode="json"),
    }
    (work_dir / "run.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )


def run_pipeline(
    spec: PipelineSpec,
    *,
    stop_at: str | None = None,
    keep_intermediates: bool = False,
) -> None:
    """Execute a pipeline end-to-end with resume support.

    ``stop_at`` — if set, stop after this stage name successfully completes.
    ``keep_intermediates`` — override ``spec.gc_mode`` to ``"keep"``.
    """
    work_dir = Path(spec.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    gc_mode = "keep" if keep_intermediates else spec.gc_mode

    run_id = _extract_run_id(spec.work_dir)  # work_dir has ${run_id} already expanded
    _write_run_snapshot(work_dir, spec, run_id)

    base_ctx = RunContext(
        work_dir=work_dir,
        pipeline_run_id=run_id,
        stage_index=0,
        stage_name="ingest",
        num_gpus=spec.num_gpus,
        num_cpu_workers=spec.num_cpu_workers or 1,
        gc_mode=gc_mode,
        device="cpu",
    )

    stage_names = [s.name for s in spec.stages]
    gc_plan = compute_gc_plan(spec)

    # Detect resume point
    last_complete = find_last_completed_stage(work_dir, stage_names)
    start_idx = 0 if last_complete is None else last_complete + 1

    if start_idx == 0:
        # Fresh run — perform ingest
        current_cuts = _run_ingest(spec, base_ctx)
    else:
        # Resume — load the last completed stage's manifest
        resume_dir = work_dir / stage_dir_name(last_complete, stage_names[last_complete])  # type: ignore[arg-type]
        logger.info("resuming from stage %s", resume_dir.name)
        current_cuts = CutSet.from_jsonl_gz(resume_dir / "cuts.jsonl.gz")

    # Execute remaining stages
    for idx in range(start_idx, len(spec.stages)):
        stage = spec.stages[idx]
        stage_ctx = replace(base_ctx, stage_index=idx, stage_name=stage.name)
        stage_dir = stage_ctx.stage_dir
        stage_dir.mkdir(parents=True, exist_ok=True)

        if is_stage_complete(stage_dir):
            logger.info("stage %s already complete, skipping", stage.name)
            current_cuts = CutSet.from_jsonl_gz(stage_dir / "cuts.jsonl.gz")
            continue

        try:
            op_cls = get_operator(stage.op)
            op_cfg = op_cls.config_cls.model_validate(stage.args)
            executor = _make_executor(op_cls.device, stage_ctx)
            current_cuts = executor.run(op_cls, op_cfg, current_cuts, stage_ctx)
        except Exception as exc:
            raise StageFailedError(stage.name, exc) from exc

        # Persist output + marker
        header = HeaderRecord(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.now(tz=timezone.utc),
            pipeline_run_id=run_id,
            stage_name=stage.name,
        )
        current_cuts.to_jsonl_gz(stage_dir / "cuts.jsonl.gz", header)
        write_success_marker(stage_dir)

        # GC after each stage
        run_gc(
            gc_plan,
            work_dir=work_dir,
            just_completed_idx=idx,
            gc_mode=gc_mode,
            stage_names=stage_names,
        )

        if stop_at == stage.name:
            logger.info("stop_at=%s reached, exiting", stop_at)
            return

    # Success — empty trash (unless keep mode)
    if gc_mode == "aggressive":
        empty_trash(work_dir)


def _run_ingest(spec: PipelineSpec, ctx: RunContext) -> CutSet:
    source_cls = get_ingest_source(spec.ingest.source)
    config = source_cls.config_cls.model_validate(spec.ingest.args)
    source = source_cls(config, ctx)
    return source.run()


def _extract_run_id(work_dir_str: str) -> str:
    """Extract the run id from the work_dir path (for resume logging).

    The work_dir path has ${run_id} already expanded by the loader; we just
    read it back. If no ``run-`` substring exists, fall back to a fresh id.
    """
    for part in Path(work_dir_str).parts:
        if part.startswith("run-"):
            return part
    from voxkitchen.utils.run_id import generate_run_id

    return generate_run_id()
```

- [ ] **Step 4: Update `src/voxkitchen/pipeline/__init__.py`**

Add to imports and `__all__`:

```python
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline
```

- [ ] **Step 5: Run the test, see it pass**

```bash
pytest tests/unit/pipeline/test_runner.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/pipeline/runner.py src/voxkitchen/pipeline/__init__.py tests/unit/pipeline/test_runner.py
git commit -m "feat(pipeline): add run_pipeline orchestration with resume and GC"
```

---

## Task 14: Wire real `vkit validate` and `vkit run` (TDD)

**Files:**
- Create: `tests/unit/cli/test_validate.py`
- Create: `tests/unit/cli/test_run.py`
- Create: `src/voxkitchen/cli/validate.py`
- Create: `src/voxkitchen/cli/run.py`
- Modify: `src/voxkitchen/cli/main.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_validate.py`:

```python
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
```

Create `tests/unit/cli/test_run.py`:

```python
"""Unit tests for the real `vkit run` command."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from voxkitchen.cli.main import app
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance


def _seed_manifest(path: Path, n: int = 3) -> None:
    cuts = [
        Cut(
            id=f"c{i}",
            recording_id="rec-1",
            start=0.0,
            duration=1.0,
            supervisions=[],
            provenance=Provenance(
                source_cut_id=None,
                generated_by="fixture",
                stage_name="00_ingest",
                created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
                pipeline_run_id="fixture",
            ),
        )
        for i in range(n)
    ]
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture",
        stage_name="00_ingest",
    )
    CutSet(cuts).to_jsonl_gz(path, header)


def test_run_completes_simple_pipeline(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _seed_manifest(input_manifest)
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: clirun
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path)])
    assert result.exit_code == 0
    assert (work_dir / "00_s0" / "_SUCCESS").exists()


def test_run_with_stop_at(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _seed_manifest(input_manifest)
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: stopat
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args: {{ path: {input_manifest} }}
stages:
  - {{ name: s0, op: identity }}
  - {{ name: s1, op: identity }}
  - {{ name: s2, op: identity }}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--stop-at", "s1"])
    assert result.exit_code == 0
    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert not (work_dir / "02_s2").exists()
```

- [ ] **Step 2: Run both tests, see them fail**

```bash
pytest tests/unit/cli/ -v
```

Expected: collection errors (modules not present).

- [ ] **Step 3: Write `src/voxkitchen/cli/validate.py`**

```python
"""Real implementation of `vkit validate`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.operators.registry import UnknownOperatorError, get_operator
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec


def validate_command(pipeline: Path) -> None:
    """Validate a pipeline YAML without executing it."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    errors: list[str] = []
    for stage in spec.stages:
        try:
            op_cls = get_operator(stage.op)
        except UnknownOperatorError as exc:
            errors.append(f"stage {stage.name!r}: {exc}")
            continue
        try:
            op_cls.config_cls.model_validate(stage.args)
        except Exception as exc:
            errors.append(f"stage {stage.name!r}: invalid args — {exc}")

    if errors:
        for e in errors:
            rprint(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)

    rprint(
        f"[green]valid[/green]: {spec.name} "
        f"({len(spec.stages)} stage(s), ingest={spec.ingest.source})"
    )
```

- [ ] **Step 4: Write `src/voxkitchen/cli/run.py`**

```python
"""Real implementation of `vkit run`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline


def run_command(
    pipeline: Path,
    num_gpus: int | None = None,
    num_workers: int | None = None,
    work_dir: str | None = None,
    resume_from: str | None = None,
    stop_at: str | None = None,
    dry_run: bool = False,
    keep_intermediates: bool = False,
) -> None:
    """Execute a pipeline."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # CLI flag overrides
    if num_gpus is not None:
        spec = spec.model_copy(update={"num_gpus": num_gpus})
    if num_workers is not None:
        spec = spec.model_copy(update={"num_cpu_workers": num_workers})
    if work_dir is not None:
        spec = spec.model_copy(update={"work_dir": work_dir})

    if dry_run:
        rprint("[yellow]--dry-run[/yellow] not yet implemented; Task 15 or later.")
        raise typer.Exit(code=0)

    try:
        run_pipeline(spec, stop_at=stop_at, keep_intermediates=keep_intermediates)
    except StageFailedError as exc:
        rprint(f"[red]stage failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    rprint("[green]pipeline complete[/green]")
```

Note: Plan 2 doesn't implement `--resume-from` explicitly because the runner's automatic resume based on `_SUCCESS` markers covers the main use case. Explicit `--resume-from <stage>` can be added later when needed.

- [ ] **Step 5: Modify `src/voxkitchen/cli/main.py`** to replace the placeholder `validate` and `run` functions

Find the existing `validate` and `run` placeholder functions and replace them with:

```python
@app.command(help="Parse and validate a pipeline YAML (no execution).")
def validate(
    pipeline: Path = typer.Argument(..., help="Pipeline YAML path."),
) -> None:
    from voxkitchen.cli.validate import validate_command

    validate_command(pipeline)


@app.command(help="Execute a pipeline.")
def run(
    pipeline: Path = typer.Argument(..., help="Pipeline YAML path."),
    num_gpus: int | None = typer.Option(None, "--num-gpus", help="Override num_gpus."),
    num_workers: int | None = typer.Option(
        None, "--num-workers", help="Override num_cpu_workers."
    ),
    work_dir: str | None = typer.Option(None, "--work-dir", help="Override work_dir."),
    resume_from: str | None = typer.Option(
        None, "--resume-from", help="Stage to resume from (not yet implemented)."
    ),
    stop_at: str | None = typer.Option(None, "--stop-at", help="Stop after this stage."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, do not execute."),
    keep_intermediates: bool = typer.Option(
        False, "--keep-intermediates", help="Disable GC; keep all derived files."
    ),
) -> None:
    from voxkitchen.cli.run import run_command

    run_command(
        pipeline=pipeline,
        num_gpus=num_gpus,
        num_workers=num_workers,
        work_dir=work_dir,
        resume_from=resume_from,
        stop_at=stop_at,
        dry_run=dry_run,
        keep_intermediates=keep_intermediates,
    )
```

Also add `from pathlib import Path` to the top of `main.py` if not already present.

The smoke test `test_placeholder_commands_exit_with_code_1` in `tests/unit/test_cli_smoke.py` tested the `init` command — `init` is still a placeholder, so the smoke test still passes.

- [ ] **Step 6: Run the tests, see them pass**

```bash
pytest tests/unit/cli/ -v
pytest tests/unit/test_cli_smoke.py -v  # existing smoke tests should still pass
```

Expected: all new CLI tests pass, smoke tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/voxkitchen/cli/validate.py src/voxkitchen/cli/run.py src/voxkitchen/cli/main.py tests/unit/cli/
git commit -m "feat(cli): wire real vkit validate and vkit run commands"
```

---

## Task 15: End-to-end integration test

**Files:**
- Create: `tests/integration/test_pipeline_e2e.py`

- [ ] **Step 1: Write the integration test**

```python
"""End-to-end integration test for the Plan 2 pipeline engine.

Uses a real YAML, a real manifest on disk, the real runner, and the
IdentityOperator across multiple stages. Verifies the full cycle:
load → ingest → stage 0 → stage 1 → stage 2 → pack as manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord, read_cuts, read_header


def test_end_to_end_pipeline_preserves_cuts(
    tmp_path: Path, sample_cutset: CutSet, sample_manifest_path: Path
) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: e2e
work_dir: {work_dir}
num_cpu_workers: 2
ingest:
  source: manifest
  args:
    path: {sample_manifest_path}
stages:
  - {{ name: pre, op: identity }}
  - {{ name: mid, op: identity }}
  - {{ name: post, op: identity }}
""",
        encoding="utf-8",
    )

    spec = load_pipeline_spec(yaml_path, run_id="run-e2e")
    run_pipeline(spec)

    # All three stages completed
    for i, name in enumerate(["pre", "mid", "post"]):
        stage_dir = work_dir / f"{i:02d}_{name}"
        assert stage_dir.exists()
        assert (stage_dir / "_SUCCESS").exists()
        assert (stage_dir / "cuts.jsonl.gz").exists()

    # Final output cuts match input
    final = list(read_cuts(work_dir / "02_post" / "cuts.jsonl.gz"))
    assert sorted(c.id for c in final) == sorted(c.id for c in sample_cutset)

    # run.yaml snapshot exists
    assert (work_dir / "run.yaml").exists()

    # Each stage's header has the right stage_name
    for i, name in enumerate(["pre", "mid", "post"]):
        h = read_header(work_dir / f"{i:02d}_{name}" / "cuts.jsonl.gz")
        assert h.stage_name == name
        assert h.schema_version == SCHEMA_VERSION


def test_end_to_end_resume_after_deleted_final_stage(
    tmp_path: Path, sample_manifest_path: Path
) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: e2e-resume
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args: {{ path: {sample_manifest_path} }}
stages:
  - {{ name: a, op: identity }}
  - {{ name: b, op: identity }}
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-resume")

    # First run
    run_pipeline(spec)
    assert (work_dir / "01_b" / "_SUCCESS").exists()

    # Simulate partial failure: delete final stage's manifest and success marker
    (work_dir / "01_b" / "_SUCCESS").unlink()
    (work_dir / "01_b" / "cuts.jsonl.gz").unlink()

    # Second run should pick up from stage b
    run_pipeline(spec)
    assert (work_dir / "01_b" / "_SUCCESS").exists()
    assert (work_dir / "01_b" / "cuts.jsonl.gz").exists()
```

- [ ] **Step 2: Run the integration tests**

```bash
pytest tests/integration/test_pipeline_e2e.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_pipeline_e2e.py
git commit -m "test(integration): add end-to-end pipeline test with identity op"
```

---

## Task 16: Full verification, lint, type, and tag

**Files:** (none — verification task)

- [ ] **Step 1: Run the entire suite**

```bash
pytest -v
```

Expected: all Plan 1 tests (40) + all Plan 2 tests pass. The new test count from Plan 2 should be approximately:
- test_run_id.py: 3
- test_base.py: 6
- test_registry.py: 8
- test_identity.py: 4
- test_spec.py: 8
- test_loader.py: 8
- test_context.py: 4
- test_checkpoint.py: 7
- test_gc.py: 10
- test_executor_cpu.py: 3
- test_executor_gpu.py: 4
- test_manifest_import.py: 3
- test_runner.py: 6
- test_validate.py: 3
- test_run.py: 2
- test_pipeline_e2e.py: 2

Total: **81 new tests**, bringing the project to **121 tests** (40 + 81).

Some test counts may be ±1 or ±2 depending on refactoring during implementation. Full suite should complete in well under 60 seconds (multiprocessing tests can add latency).

- [ ] **Step 2: Run ruff check + format**

```bash
ruff check src tests
ruff format --check src tests
```

Both clean. Fix with `ruff format` if needed and commit as `style: ruff format fixes` if there are any changes.

- [ ] **Step 3: Run mypy strict**

```bash
mypy src/voxkitchen tests
```

Expected: `Success: no issues found in N source files`. The pre-existing "unused section" note for soundfile/yaml is harmless.

- [ ] **Step 4: Verify CLI still works**

```bash
vkit --help
vkit validate --help
vkit run --help
```

All three should print help cleanly.

- [ ] **Step 5: Verify the placeholder smoke test still passes**

```bash
pytest tests/unit/test_cli_smoke.py -v
```

Expected: 3 passed. `init`, `ingest`, and `viz` are still placeholders; `validate` and `run` are real but the smoke test only tests their registration.

- [ ] **Step 6: Run pre-commit on everything**

```bash
pre-commit run --all-files
```

Expected: all hooks pass.

- [ ] **Step 7: Tag Plan 2 completion**

```bash
git tag -a plan-02-pipeline-engine -m "Plan 2 complete: pipeline engine with runner, executors, GC, CLI wiring"
git tag -l
```

Expected: both `plan-01-foundation` and `plan-02-pipeline-engine` tags exist.

---

## Plan 2 Completion Checklist

Before declaring Plan 2 complete, verify every item:

- [ ] `vkit validate examples/pipelines/minimal.yaml` accepts a valid identity-only pipeline (create `examples/pipelines/minimal.yaml` as part of this item if it doesn't exist — same content as one of the test YAMLs)
- [ ] `vkit run pipeline.yaml` successfully runs a 3-stage identity pipeline end-to-end against a manifest ingest source
- [ ] `vkit run --stop-at` honors the stop point
- [ ] `vkit run --keep-intermediates` disables GC
- [ ] A resumed pipeline (delete stage N's `_SUCCESS`, re-run) picks up from stage N, not stage 0
- [ ] CpuPoolExecutor works with `num_workers=1`, `num_workers=2`, and empty CutSet input
- [ ] GpuPoolExecutor runs an identity-like operator on a CPU-only machine (tests verify CUDA_VISIBLE_DEVICES is set per worker)
- [ ] `vkit validate` reports unknown operators and malformed YAML with exit code 1
- [ ] `vkit run` reports stage failures with exit code 2
- [ ] `work_dir/run.yaml` contains a snapshot of the expanded pipeline spec with the run id
- [ ] Every stage directory follows the `NN_<stage_name>` naming convention
- [ ] `derived_trash/` is emptied after successful runs in aggressive mode
- [ ] All 121 unit + integration tests pass
- [ ] `ruff check`, `ruff format --check`, and `mypy src/voxkitchen tests` all clean
- [ ] Every new Python file has `from __future__ import annotations` at the top
- [ ] `operators/base.py` uses `TYPE_CHECKING` for `RunContext` (no circular import)
- [ ] `ingest/base.py` uses `TYPE_CHECKING` for `RunContext`
- [ ] `git tag plan-02-pipeline-engine` placed at HEAD
- [ ] Plan 3 can start without touching any file in `operators/base.py`, `pipeline/spec.py`, `pipeline/loader.py`, `pipeline/context.py`, `pipeline/checkpoint.py`, `pipeline/gc.py`, or `pipeline/runner.py` — those are "done"

## What Plan 3 Will Build On

After Plan 2 is tagged, Plan 3 will:
- Add `DirScanIngestSource` (`source: dir`) and the recipe framework
- Add 4 basic-category operators: `ffmpeg_convert`, `resample`, `channel_merge`, `loudness_normalize`
- Add `pack_manifest` (trivial, writes the CutSet as-is)
- Not touch runner, executors, schema, registry, spec, loader, context, or checkpoint

Plan 3's first real test will be a pipeline like:
```yaml
ingest: { source: dir, args: { root: /tmp/wavs } }
stages:
  - { name: convert, op: ffmpeg_convert, args: { target_format: wav } }
  - { name: resample, op: resample, args: { target_sr: 16000 } }
  - { name: pack, op: pack_manifest, args: { output: /tmp/out.jsonl.gz } }
```

which exercises the `produces_audio=True` path in GC for the first time.
