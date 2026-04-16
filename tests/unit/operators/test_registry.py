"""Unit tests for voxkitchen.operators.registry."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import (
    _REGISTRY,
    MissingExtrasError,
    UnknownOperatorError,
    get_operator,
    list_operators,
    register_operator,
)
from voxkitchen.schema.cutset import CutSet


class _TestConfig(OperatorConfig):
    x: int = 0


@pytest.fixture(autouse=True)
def _clear_registry() -> Generator[None, None, None]:
    """Each test runs against a clean registry snapshot."""
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _make_op(op_name: str) -> type[Operator]:
    class _DynOp(Operator):
        name = op_name
        config_cls = _TestConfig

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    return _DynOp


def test_register_operator_adds_to_registry() -> None:
    op_cls = _make_op("alpha")
    register_operator(op_cls)
    assert "alpha" in _REGISTRY
    assert _REGISTRY["alpha"] is op_cls


def test_register_operator_returns_the_class_for_decorator_use() -> None:
    op_cls = _make_op("beta")
    returned = register_operator(op_cls)
    assert returned is op_cls


def test_register_operator_rejects_duplicate_names() -> None:
    register_operator(_make_op("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_operator(_make_op("dup"))


def test_register_operator_rejects_empty_name() -> None:
    class _Empty(Operator):
        name = ""
        config_cls = _TestConfig

        def process(self, cuts: CutSet) -> CutSet:
            return cuts

    with pytest.raises(ValueError, match="must declare"):
        register_operator(_Empty)


def test_get_operator_returns_registered_class() -> None:
    op_cls = _make_op("gamma")
    register_operator(op_cls)
    assert get_operator("gamma") is op_cls


def test_get_operator_raises_on_unknown_name() -> None:
    register_operator(_make_op("delta"))
    with pytest.raises(UnknownOperatorError) as exc_info:
        get_operator("deltaa")  # typo
    # Suggestion should include the close match
    assert "delta" in exc_info.value.suggestions


def test_list_operators_returns_sorted_names() -> None:
    for n in ["zulu", "alpha", "mike"]:
        register_operator(_make_op(n))
    assert list_operators() == ["alpha", "mike", "zulu"]


def test_missing_extras_error_message_includes_pip_hint() -> None:
    err = MissingExtrasError("faster_whisper_asr", ["asr"])
    msg = str(err)
    assert "faster_whisper_asr" in msg
    assert "pip install voxkitchen[asr]" in msg
