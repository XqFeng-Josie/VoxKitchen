"""Operator ABC and OperatorConfig base.

Every VoxKitchen operator subclasses ``Operator`` and declares:

1. A ``name`` (string, unique across the registry).
2. A ``config_cls`` subclass of ``OperatorConfig`` describing its parameters.
3. A ``process(cuts) -> cuts`` implementation.

Operators may override ``setup()`` and ``teardown()`` for model loading/release.
They may also override the class variables ``device`` (``"cpu"`` | ``"gpu"``),
``parallelizable`` (whether the executor may shard the input CutSet across
workers; set this to ``False`` for batch exporters that write one shared output
directory), ``produces_audio`` (whether the operator creates new audio files on
disk), ``reads_audio_bytes`` (whether downstream stages need to read audio
samples from this stage's outputs), and ``required_extras`` (names of pyproject
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

# The third-party operator API version. Bump the major component only on a
# BREAKING change to the operator contract (process() signature, removal/rename
# of a required ClassVar, or a change to field-contract token semantics).
# Adding new *optional* ClassVars with safe defaults is backward-compatible and
# does NOT bump this. Informational: not enforced at runtime.
OPERATOR_API_VERSION = 1


class OperatorConfig(BaseModel):
    """Base class for operator parameter models. Forbids unknown fields."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


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
    parallelizable: ClassVar[bool] = True
    produces_audio: ClassVar[bool] = False
    reads_audio_bytes: ClassVar[bool] = True
    required_extras: ClassVar[list[str]] = []

    # ---- Field contract (Workstream A) ---------------------------------
    # Declarative I/O so the pre-flight validator can reject broken chains
    # before execution. Tokens come from a fixed field vocabulary
    # (audio, supervisions.text/language/speaker/gender, metrics.<name>,
    # custom.<key>, and namespace wildcards like metrics.* used in clears).
    reads: ClassVar[list[str]] = []  # required inputs; missing -> preflight error
    writes: ClassVar[list[str]] = []  # fields this op produces
    optional_reads: ClassVar[list[str]] = []  # used-if-present; missing -> warning
    clears: ClassVar[list[str]] = []  # fields reset by this op (e.g. VAD re-segment)
    contract_exempt: ClassVar[bool] = False  # opt out of the completeness meta-test

    def __init__(self, config: OperatorConfig, ctx: RunContext) -> None:
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

    def dynamic_reads(self) -> list[str]:
        """Extra required-read tokens derived from this operator's config.

        Override for operators whose inputs depend on YAML args (e.g. a filter
        whose conditions reference ``metrics.snr``). Default: none.
        """
        return []

    def teardown(self) -> None:
        """Called once per worker process before the worker exits.

        Override to release GPU memory, close file handles, etc.
        """
        return None
