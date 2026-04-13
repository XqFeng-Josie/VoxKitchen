"""RecipeIngestSource: dispatch to a named recipe for dataset-specific ingestion."""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.ingest.recipes import get_recipe
from voxkitchen.schema.cutset import CutSet


class RecipeConfig(IngestConfig):
    recipe: str
    root: str
    subsets: list[str] | None = None


class RecipeIngestSource(IngestSource):
    name = "recipe"
    config_cls = RecipeConfig

    def run(self) -> CutSet:
        assert isinstance(self.config, RecipeConfig)
        recipe = get_recipe(self.config.recipe)
        return recipe.prepare(
            root=Path(self.config.root),
            subsets=self.config.subsets,
            ctx=self.ctx,
        )
