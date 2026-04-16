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
