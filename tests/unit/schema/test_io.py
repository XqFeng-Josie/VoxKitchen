"""Unit tests for voxkitchen.schema.io: JSONL.gz reading/writing with header."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.io import (
    SCHEMA_VERSION,
    HeaderRecord,
    IncompatibleSchemaError,
    read_cuts,
    read_header,
    write_cuts,
)
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.supervision import Supervision


def _make_cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=3.5,
        supervisions=[Supervision(id=f"{cid}-sup", recording_id="rec-1", start=0.0, duration=3.5)],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def test_write_then_read_round_trips_all_cuts(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    cuts = [_make_cut(f"cut-{i}") for i in range(3)]
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter(cuts))
    restored = list(read_cuts(path))
    assert restored == cuts


def test_header_is_first_line_of_file(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter([_make_cut("cut-1")]))
    with gzip.open(path, "rt") as f:
        first_line = f.readline()
    parsed = json.loads(first_line)
    assert parsed["__type__"] == "voxkitchen.header"
    assert parsed["schema_version"] == SCHEMA_VERSION


def test_read_header_returns_header_without_consuming_cuts(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="02_vad",
    )
    write_cuts(path, header, iter([_make_cut("cut-1")]))
    read_back = read_header(path)
    assert read_back.stage_name == "02_vad"
    assert read_back.pipeline_run_id == "run-a1b2c3"


def test_incompatible_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "cuts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "__type__": "voxkitchen.header",
                    "schema_version": "99.0",
                    "created_at": "2026-04-11T10:30:00+00:00",
                    "pipeline_run_id": "run-x",
                    "stage_name": "00_ingest",
                }
            )
            + "\n"
        )
    with pytest.raises(IncompatibleSchemaError):
        list(read_cuts(path))


def test_read_cuts_on_header_only_manifest_returns_empty_iterator(tmp_path: Path) -> None:
    """A manifest with a valid header and zero cut lines is well-formed."""
    path = tmp_path / "cuts.jsonl.gz"
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )
    write_cuts(path, header, iter([]))
    assert list(read_cuts(path)) == []


def test_read_cuts_on_zero_byte_gzip_raises(tmp_path: Path) -> None:
    """A zero-byte gzip stream (e.g. from a crashed writer) is not a legal manifest."""
    path = tmp_path / "crashed.jsonl.gz"
    with gzip.open(path, "wb") as f:
        f.write(b"")
    with pytest.raises(IncompatibleSchemaError):
        list(read_cuts(path))


def test_read_header_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    """read_header must validate schema_version, matching read_cuts."""
    path = tmp_path / "future.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "__type__": "voxkitchen.header",
                    "schema_version": "99.0",
                    "created_at": "2026-04-11T10:30:00+00:00",
                    "pipeline_run_id": "run-x",
                    "stage_name": "00_ingest",
                }
            )
            + "\n"
        )
    with pytest.raises(IncompatibleSchemaError):
        read_header(path)
