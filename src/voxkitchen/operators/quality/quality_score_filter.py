"""QualityScoreFilter operator: keep Cuts that satisfy all expression conditions."""

from __future__ import annotations

import ast
import operator as op_mod
from typing import Any

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet

_OPS = {
    ">": op_mod.gt,
    ">=": op_mod.ge,
    "<": op_mod.lt,
    "<=": op_mod.le,
    "==": op_mod.eq,
    "!=": op_mod.ne,
}


def _resolve_field(cut: Cut, path: str) -> Any:
    obj: Any = cut
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj[part]
        else:
            obj = getattr(obj, part)
    return obj


def _eval_condition(condition: str, cut: Cut) -> bool:
    parts = condition.split()
    if len(parts) != 3:
        raise ValueError(f"condition must be 'field op value', got: {condition!r}")
    field_path, op_str, raw_value = parts
    if op_str not in _OPS:
        raise ValueError(f"unsupported operator: {op_str!r}")
    value = ast.literal_eval(raw_value)
    actual = _resolve_field(cut, field_path)
    return bool(_OPS[op_str](actual, value))


class QualityScoreFilterConfig(OperatorConfig):
    conditions: list[str]  # e.g. ["metrics.snr > 10", "duration > 0.5"]


@register_operator
class QualityScoreFilterOperator(Operator):
    """Drop Cuts that do not satisfy all conditions.

    Each condition is a whitespace-separated triple ``field.path op value``
    where ``op`` is one of ``>``, ``>=``, ``<``, ``<=``, ``==``, ``!=``.
    All conditions are AND-ed together.
    """

    name = "quality_score_filter"
    config_cls = QualityScoreFilterConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, QualityScoreFilterConfig)
        return CutSet(
            cut
            for cut in cuts
            if all(_eval_condition(cond, cut) for cond in self.config.conditions)
        )
