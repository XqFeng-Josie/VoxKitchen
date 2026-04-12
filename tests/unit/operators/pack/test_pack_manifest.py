"""Unit tests for pack_manifest operator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.pack.pack_manifest import PackManifestConfig, PackManifestOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
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
            generated_by="test",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_pack_manifest_is_registered() -> None:
    assert get_operator("pack_manifest") is PackManifestOperator


def test_pack_manifest_does_not_produce_audio() -> None:
    assert PackManifestOperator.produces_audio is False
    assert PackManifestOperator.reads_audio_bytes is False


def test_pack_manifest_passes_cuts_through() -> None:
    cs = CutSet([_cut("c0"), _cut("c1"), _cut("c2")])
    op = PackManifestOperator(PackManifestConfig(), ctx=object())  # type: ignore[arg-type]
    result = op.process(cs)
    assert [c.id for c in result] == ["c0", "c1", "c2"]
