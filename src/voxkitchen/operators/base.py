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

    def teardown(self) -> None:
        """Called once per worker process before the worker exits.

        Override to release GPU memory, close file handles, etc.
        """
        return None
