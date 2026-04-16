"""DurationFilter operator: keep only Cuts within a duration range."""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class DurationFilterConfig(OperatorConfig):
    min_duration: float = 0.0
    max_duration: float | None = None  # None means no upper bound


@register_operator
class DurationFilterOperator(Operator):
    """Drop Cuts whose duration falls outside [min_duration, max_duration].

    This is an N-to-fewer operator: no audio is read or written.
    """

    name = "duration_filter"
    config_cls = DurationFilterConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, DurationFilterConfig)
        lo = self.config.min_duration
        hi = self.config.max_duration
        if hi is None:
            return CutSet(c for c in cuts if lo <= c.duration)
        return CutSet(c for c in cuts if lo <= c.duration <= hi)
