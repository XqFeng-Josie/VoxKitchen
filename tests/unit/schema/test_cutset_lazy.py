"""Unit tests for CutSet lazy loading mode."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str, duration: float = 1.0) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=duration,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def _header() -> HeaderRecord:
    return HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name="00_ingest",
    )


def _write_manifest(tmp_path: Path, n: int = 5) -> Path:
    """Write a manifest with n cuts and return its path."""
    path = tmp_path / "cuts.jsonl.gz"
    cs = CutSet([_cut(f"c{i}", duration=float(i + 1)) for i in range(n)])
    cs.to_jsonl_gz(path, _header())
    return path


def test_lazy_from_jsonl_gz_does_not_materialize(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    assert cs.is_lazy


def test_lazy_iteration_streams_without_materializing(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=3)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    ids = [c.id for c in cs]
    assert ids == ["c0", "c1", "c2"]
    # Still lazy after iteration (did not materialize into list)
    assert cs.is_lazy


def test_lazy_iteration_can_be_repeated(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=2)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    first = [c.id for c in cs]
    second = [c.id for c in cs]
    assert first == second == ["c0", "c1"]


def test_lazy_len_triggers_materialization(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=4)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    assert cs.is_lazy
    assert len(cs) == 4
    assert not cs.is_lazy


def test_lazy_split_triggers_materialization(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=6)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    shards = cs.split(3)
    assert not cs.is_lazy
    all_ids = sorted(c.id for shard in shards for c in shard)
    assert all_ids == [f"c{i}" for i in range(6)]


def test_lazy_filter_returns_eager_cutset(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=5)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    filtered = cs.filter(lambda c: c.duration >= 3.0)
    # filtered is eager (materialized), source stays lazy
    assert not filtered.is_lazy
    assert [c.id for c in filtered] == ["c2", "c3", "c4"]


def test_lazy_map_returns_eager_cutset(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=2)
    cs = CutSet.from_jsonl_gz(path, lazy=True)
    mapped = cs.map(lambda c: c.model_copy(update={"metrics": {"x": 1.0}}))
    assert not mapped.is_lazy
    assert all(c.metrics["x"] == 1.0 for c in mapped)


def test_lazy_to_jsonl_gz_streams_without_materializing(tmp_path: Path) -> None:
    src = _write_manifest(tmp_path, n=3)
    cs = CutSet.from_jsonl_gz(src, lazy=True)
    dst = tmp_path / "copy.jsonl.gz"
    cs.to_jsonl_gz(dst, _header())
    assert cs.is_lazy  # did not materialize
    restored = CutSet.from_jsonl_gz(dst)
    assert [c.id for c in restored] == ["c0", "c1", "c2"]


def test_eager_cutset_is_not_lazy() -> None:
    cs = CutSet([_cut("c0")])
    assert not cs.is_lazy


def test_from_jsonl_gz_default_is_eager(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, n=2)
    cs = CutSet.from_jsonl_gz(path)
    assert not cs.is_lazy
    assert len(cs) == 2
