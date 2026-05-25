"""Every registered operator must declare a field contract (or opt out)."""
from __future__ import annotations

import voxkitchen.operators  # noqa: F401  (populates the registry)
from voxkitchen.operators.registry import get_operator, list_operators


def test_every_operator_declares_a_contract():
    missing: list[str] = []
    for name in list_operators():
        op_cls = get_operator(name)
        if op_cls.contract_exempt:
            continue
        if not (op_cls.reads or op_cls.writes or op_cls.optional_reads or op_cls.clears):
            missing.append(name)
    assert not missing, f"operators without a contract: {sorted(missing)}"
