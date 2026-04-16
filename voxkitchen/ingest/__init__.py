"""VoxKitchen ingest sources: how Cuts enter a pipeline."""

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.ingest.dir_scan import DirScanConfig, DirScanIngestSource
from voxkitchen.ingest.manifest_import import (
    ManifestIngestConfig,
    ManifestIngestSource,
)
from voxkitchen.ingest.recipe_source import RecipeConfig, RecipeIngestSource

# Registry of ingest sources keyed by the IngestSpec.source literal
_INGEST_SOURCES: dict[str, type[IngestSource]] = {
    "dir": DirScanIngestSource,
    "manifest": ManifestIngestSource,
    "recipe": RecipeIngestSource,
}


def get_ingest_source(name: str) -> type[IngestSource]:
    if name not in _INGEST_SOURCES:
        raise KeyError(
            f"ingest source {name!r} not available in this build. "
            f"Available: {sorted(_INGEST_SOURCES.keys())}"
        )
    return _INGEST_SOURCES[name]


__all__ = [
    "DirScanConfig",
    "DirScanIngestSource",
    "IngestConfig",
    "IngestSource",
    "ManifestIngestConfig",
    "ManifestIngestSource",
    "RecipeConfig",
    "RecipeIngestSource",
    "get_ingest_source",
]
