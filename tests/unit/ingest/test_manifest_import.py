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

    ingest = ManifestIngestSource(ManifestIngestConfig(path=str(src_path)), ctx=_ctx(tmp_path))
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
