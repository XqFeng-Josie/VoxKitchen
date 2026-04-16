"""IdentityOperator: the simplest possible operator (no-op passthrough).

Ships with Plan 2 primarily as a test scaffold. It has no parameters, no
dependencies, and does not touch audio bytes — so integration tests can
verify pipeline orchestration end-to-end without needing ffmpeg, torch,
or any real data.
"""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class IdentityConfig(OperatorConfig):
    """IdentityOperator has no tunable parameters."""


@register_operator
class IdentityOperator(Operator):
    """Pass cuts through unchanged (no-op, useful for testing)."""

    name = "identity"
    config_cls = IdentityConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False  # identity never touches samples

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(list(cuts))
