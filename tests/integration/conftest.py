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
