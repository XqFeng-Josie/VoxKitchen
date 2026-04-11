"""Unit tests for voxkitchen.schema.cutset.CutSet."""

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


def _header(stage: str = "00_ingest") -> HeaderRecord:
    return HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
        stage_name=stage,
    )


def test_cutset_len_and_iter() -> None:
    cs = CutSet([_cut("c0"), _cut("c1"), _cut("c2")])
    assert len(cs) == 3
    ids = [c.id for c in cs]
    assert ids == ["c0", "c1", "c2"]


def test_cutset_split_into_n_shards_balanced() -> None:
    cs = CutSet([_cut(f"c{i}") for i in range(10)])
    shards = cs.split(3)
    assert len(shards) == 3
    # No Cut lost, no duplication
    all_ids = [c.id for shard in shards for c in shard]
    assert sorted(all_ids) == sorted(c.id for c in cs)
    # Sizes within 1 of each other
    sizes = [len(s) for s in shards]
    assert max(sizes) - min(sizes) <= 1


def test_cutset_split_n_larger_than_len_produces_empty_shards() -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])
    shards = cs.split(5)
    assert len(shards) == 5
    assert sum(len(s) for s in shards) == 2


def test_cutset_filter_preserves_matching_cuts_only() -> None:
    cs = CutSet([_cut("c0", 1.0), _cut("c1", 5.0), _cut("c2", 3.0)])
    filtered = cs.filter(lambda c: c.duration >= 3.0)
    assert [c.id for c in filtered] == ["c1", "c2"]


def test_cutset_map_applies_transformation() -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])

    def add_metric(c: Cut) -> Cut:
        return c.model_copy(update={"metrics": {"snr": 20.0}})

    mapped = cs.map(add_metric)
    assert all(c.metrics["snr"] == 20.0 for c in mapped)


def test_cutset_merge_concatenates_all_cuts() -> None:
    a = CutSet([_cut("a0"), _cut("a1")])
    b = CutSet([_cut("b0")])
    c = CutSet([_cut("c0"), _cut("c1")])
    merged = CutSet.merge([a, b, c])
    assert [x.id for x in merged] == ["a0", "a1", "b0", "c0", "c1"]


def test_cutset_to_and_from_jsonl_gz_round_trips(tmp_path: Path) -> None:
    cs = CutSet([_cut("c0"), _cut("c1")])
    path = tmp_path / "cuts.jsonl.gz"
    cs.to_jsonl_gz(path, _header())
    restored = CutSet.from_jsonl_gz(path)
    assert [c.id for c in restored] == ["c0", "c1"]


def test_cutset_concat_from_disk_joins_shards_in_order(tmp_path: Path) -> None:
    paths = []
    for i in range(3):
        p = tmp_path / f"shard_{i}.jsonl.gz"
        CutSet([_cut(f"s{i}c0"), _cut(f"s{i}c1")]).to_jsonl_gz(p, _header())
        paths.append(p)
    merged = CutSet.concat_from_disk(paths)
    assert [c.id for c in merged] == [
        "s0c0",
        "s0c1",
        "s1c0",
        "s1c1",
        "s2c0",
        "s2c1",
    ]
