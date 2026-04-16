"""Unit tests for voxkitchen.operators.base."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError
from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.schema.cutset import CutSet

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext  # noqa: F401


class DemoConfig(OperatorConfig):
    threshold: float = 0.5


class DemoOperator(Operator):
    name = "demo"
    config_cls = DemoConfig

    def process(self, cuts: CutSet) -> CutSet:
        return cuts


def test_operator_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DemoConfig.model_validate({"threshold": 0.5, "surprise": "boom"})


def test_operator_config_applies_defaults() -> None:
    cfg = DemoConfig()
    assert cfg.threshold == 0.5


def test_operator_instantiates_with_config_and_ctx() -> None:
    cfg = DemoConfig(threshold=0.9)
    # ctx is not used at construction time; pass a sentinel
    op = DemoOperator(cfg, ctx=object())  # type: ignore[arg-type]
    assert cast(DemoConfig, op.config).threshold == 0.9
    assert op.ctx is not None


def test_operator_setup_and_teardown_are_noops_by_default() -> None:
    cfg = DemoConfig()
    op = DemoOperator(cfg, ctx=object())  # type: ignore[arg-type]
    # should not raise
    op.setup()
    op.teardown()


def test_operator_class_attributes_have_sensible_defaults() -> None:
    assert DemoOperator.device == "cpu"
    assert DemoOperator.produces_audio is False
    assert DemoOperator.reads_audio_bytes is True
    assert DemoOperator.required_extras == []


def test_abstract_process_enforced_by_abc() -> None:
    class IncompleteOperator(Operator):
        name = "incomplete"
        config_cls = DemoConfig

    with pytest.raises(TypeError):
        IncompleteOperator(DemoConfig(), ctx=object())  # type: ignore[abstract, arg-type]
