"""Recipe base class: parse a dataset directory into a CutSet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


class Recipe(ABC):
    name: str = ""

    @abstractmethod
    def prepare(self, root: Path, subsets: list[str] | None, ctx: RunContext) -> CutSet:
        """Parse a locally-present dataset and return a CutSet."""
