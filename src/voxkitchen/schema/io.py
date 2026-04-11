"""Serialization of Cuts to and from ``cuts.jsonl.gz`` manifests.

Format (one JSON object per line, gzip-compressed):

    {"__type__": "voxkitchen.header", "schema_version": "0.1", ...}
    {"__type__": "cut", "id": "...", "recording_id": "...", ...}
    {"__type__": "cut", ...}
    ...

The header record enables future schema migrations: readers that encounter
an unknown version can be routed through a migration function.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.cut import Cut

SCHEMA_VERSION = "0.1"


class IncompatibleSchemaError(RuntimeError):
    """Raised when a manifest's schema version is not understood."""


class HeaderRecord(BaseModel):
    """First-line metadata record in every ``cuts.jsonl.gz`` file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    created_at: datetime
    pipeline_run_id: str
    stage_name: str


def write_cuts(path: Path, header: HeaderRecord, cuts: Iterable[Cut]) -> None:
    """Write a header followed by a stream of Cuts to a gzipped JSONL file.

    Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        header_obj = {"__type__": "voxkitchen.header", **header.model_dump(mode="json")}
        f.write(json.dumps(header_obj, ensure_ascii=False) + "\n")
        for cut in cuts:
            cut_obj = {"__type__": "cut", **cut.model_dump(mode="json")}
            f.write(json.dumps(cut_obj, ensure_ascii=False) + "\n")


def read_header(path: Path) -> HeaderRecord:
    """Return the header record of a manifest without reading the rest."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline()
    if not first:
        raise IncompatibleSchemaError(f"manifest is empty: {path}")
    parsed = json.loads(first)
    if parsed.get("__type__") != "voxkitchen.header":
        raise IncompatibleSchemaError(
            f"expected header on first line of {path}, got {parsed.get('__type__')!r}"
        )
    payload = {k: v for k, v in parsed.items() if k != "__type__"}
    return HeaderRecord.model_validate(payload)


def read_cuts(path: Path) -> Iterator[Cut]:
    """Yield Cuts from a manifest, validating the header first.

    Raises ``IncompatibleSchemaError`` if the file has no header or the
    schema version does not match ``SCHEMA_VERSION``.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        parsed = json.loads(first)
        if parsed.get("__type__") != "voxkitchen.header":
            raise IncompatibleSchemaError(
                f"expected header on first line of {path}, got {parsed.get('__type__')!r}"
            )
        if parsed.get("schema_version") != SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                f"schema version mismatch in {path}: "
                f"file is {parsed.get('schema_version')!r}, "
                f"reader supports {SCHEMA_VERSION!r}"
            )
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("__type__") != "cut":
                continue
            cut_payload = {k: v for k, v in obj.items() if k != "__type__"}
            yield Cut.model_validate(cut_payload)
