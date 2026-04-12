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

    def __init__(self, config: IngestConfig, ctx: RunContext) -> None:
        self.config = config
        self.ctx = ctx

    @abstractmethod
    def run(self) -> CutSet:
        """Produce the initial CutSet for a pipeline."""
