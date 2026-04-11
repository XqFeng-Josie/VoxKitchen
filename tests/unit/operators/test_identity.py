"""Unit tests for voxkitchen.operators.noop.identity.IdentityOperator."""

from __future__ import annotations

from datetime import datetime, timezone

from voxkitchen.operators.noop.identity import IdentityConfig, IdentityOperator
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
            generated_by="test@0.0",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        ),
    )


def test_identity_operator_is_registered_as_identity() -> None:
    assert get_operator("identity") is IdentityOperator


def test_identity_operator_is_cpu_and_noop() -> None:
    assert IdentityOperator.device == "cpu"
    assert IdentityOperator.produces_audio is False
    assert IdentityOperator.required_extras == []


def test_identity_process_returns_equivalent_cutset() -> None:
    cs = CutSet([_cut("c0"), _cut("c1"), _cut("c2")])
    op = IdentityOperator(IdentityConfig(), ctx=object())
    result = op.process(cs)
    assert [c.id for c in result] == ["c0", "c1", "c2"]


def test_identity_config_has_no_required_fields() -> None:
    cfg = IdentityConfig()
    assert cfg is not None
