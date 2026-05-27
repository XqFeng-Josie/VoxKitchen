# Operator Plugins

A third-party operator plugin is a regular pip-installable Python package
that exposes one or more operators to VoxKitchen through the
`voxkitchen.operators` [entry-point group][ep].  VoxKitchen discovers plugins
lazily at runtime — no source changes to the core are required.

[ep]: https://packaging.python.org/en/latest/specifications/entry-points/

---

## 1. What a plugin is

When VoxKitchen's operator registry is first accessed (on `vkit operators`,
`vkit doctor`, `vkit run`, etc.) it calls
`importlib.metadata.entry_points(group="voxkitchen.operators")` and loads
every advertised class into the same registry that holds the built-in
operators.  Any operator name that is unique and satisfies the base contract
is usable in `pipeline.yaml` stages exactly like a built-in.

---

## 2. Write the operator

Subclass `voxkitchen.operators.Operator` and satisfy three requirements:

1. Set `name` to a unique string (the key used in `pipeline.yaml`).
2. Set `config_cls` to a subclass of `OperatorConfig` (a Pydantic model
   describing the operator's YAML-configurable parameters).
3. Implement `process(self, cuts: CutSet) -> CutSet`.

Declare the field contract with four `ClassVar` lists (all required; the
pre-flight validator checks them before a run starts):

| ClassVar | Meaning |
|----------|---------|
| `reads` | Fields that must be present on each cut. |
| `writes` | Fields this operator populates or updates. |
| `optional_reads` | Fields consumed when present; emits a warning if absent. |
| `clears` | Fields this operator removes. |

See [Field Contracts](../architecture.md#field-contracts) for the full token
vocabulary (`audio`, `supervisions.text`, `metrics.<name>`, `custom.<key>`,
namespace wildcards, etc.).

```python
from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.schema.cutset import CutSet


class WordCountConfig(OperatorConfig):
    """No parameters."""


class WordCountOperator(Operator):
    """Count words in each cut's first transcribed supervision.

    Writes the count to ``metrics.word_count``.
    """

    name = "word_count"
    config_cls = WordCountConfig

    # Execution ClassVars (base-class defaults shown; override as needed)
    device = "cpu"              # "cpu" | "gpu"
    produces_audio = False      # creates new audio files on disk?
    reads_audio_bytes = True    # default True; set False if your op ignores audio
    required_extras: ClassVar[list[str]] = []
    parallelizable: ClassVar[bool] = True

    # Field contract
    reads: ClassVar[list[str]] = ["supervisions.text"]
    writes: ClassVar[list[str]] = ["metrics.word_count"]
    optional_reads: ClassVar[list[str]] = []
    clears: ClassVar[list[str]] = []

    def process(self, cuts: CutSet) -> CutSet:
        out = []
        for cut in cuts:
            text = next((s.text for s in cut.supervisions if s.text), "")
            metrics = {**cut.metrics, "word_count": float(len(text.split()))}
            out.append(cut.model_copy(update={"metrics": metrics}))
        return CutSet(out)
```

!!! warning "Do NOT use `@register_operator`"
    The `@register_operator` decorator is the built-in registration mechanism
    (used inside `voxkitchen/operators/`).  For plugins the entry point **is**
    the registration mechanism — applying the decorator as well would
    double-register the operator and raise a `ValueError` ("already registered").

---

## 3. Package it

A minimal `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "voxkitchen-my-plugin"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["voxkitchen"]

[project.entry-points."voxkitchen.operators"]
word_count = "voxkitchen_my_plugin.operator:WordCountOperator"

[tool.hatch.build.targets.wheel]
packages = ["voxkitchen_my_plugin"]
```

The entry-point key (`word_count` above) is only a label; the registry uses
the `name` ClassVar on the operator class.  Use a descriptive label that
matches your operator name to avoid confusion.

You may register multiple operators from a single package by adding one entry
per operator under `[project.entry-points."voxkitchen.operators"]`.

---

## 4. Install and verify

```bash
pip install ./voxkitchen-my-plugin    # or any pip-installable path / URL

# The operator now appears in the list
vkit operators

# Third-party operators are listed under an "Other" category unless their
# module path matches a built-in category prefix.
vkit operators show word_count

# doctor reports the operator API version and the number of third-party
# operators discovered in the active environment
vkit doctor
```

---

## 5. Stable surface and compatibility

Plugins may depend on the following elements of the operator contract:

- **Identity**: `name`, `config_cls`
- **Runtime method**: `process(cuts: CutSet) -> CutSet`
- **Execution ClassVars**: `device`, `required_extras`, `produces_audio`,
  `reads_audio_bytes`, `parallelizable`
- **Field-contract ClassVars**: `reads`, `writes`, `optional_reads`, `clears`
- **API version constant**:
  `from voxkitchen.operators import OPERATOR_API_VERSION` (currently `1`)

**Compatibility policy:**

- New *optional* ClassVars with safe defaults (e.g. a new execution hint)
  are backward-compatible and do not bump `OPERATOR_API_VERSION`.
- Renaming or removing existing ClassVars, changing the signature of
  `process()`, or any other breaking change bumps the *major* version of
  VoxKitchen and increments `OPERATOR_API_VERSION`.

If your package needs to guard against a future breaking change, check:

```python
from voxkitchen.operators import OPERATOR_API_VERSION
assert OPERATOR_API_VERSION == 1, f"Unsupported API version: {OPERATOR_API_VERSION}"
```

---

## 6. Worked example

A complete, installable example is at
[`examples/plugin-operator/`](../../examples/plugin-operator/).  It
implements the `word_count` operator shown above and is structured exactly
as a real third-party package would be.

---

## 7. Current limitation

A plugin runs in the Python environment where it is installed.  The
multi-environment Docker images (`asr`, `diarize`, `tts`, etc.) do not yet
automatically map a third-party operator to a specific inner env — the
runner does not know which env satisfies the plugin's dependencies.

**Today:** single-environment local `vkit run` and container-internal
`vkit run` both work fine as long as the plugin is installed in the active
environment.

**Workaround for multi-env images:** install the plugin inside the target env
in your Dockerfile and set `required_extras` to an existing extras group that
maps to the desired env via `EXTRA_TO_ENV` in
`voxkitchen/runtime/env_resolver.py`.  Full first-class multi-env dispatch for
third-party operators is planned.
