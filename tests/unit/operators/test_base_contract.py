"""Contract attributes on the Operator base class."""
from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.schema.cutset import CutSet


class _Dummy(Operator):
    name = "dummy_contract_test"
    config_cls = OperatorConfig

    def process(self, cuts: CutSet) -> CutSet:  # pragma: no cover
        return cuts


def test_contract_attrs_default_empty():
    assert _Dummy.reads == []
    assert _Dummy.writes == []
    assert _Dummy.optional_reads == []
    assert _Dummy.clears == []
    assert _Dummy.contract_exempt is False


def test_dynamic_reads_default_empty():
    op = _Dummy(OperatorConfig(), ctx=None)  # ctx unused by default hook
    assert op.dynamic_reads() == []
