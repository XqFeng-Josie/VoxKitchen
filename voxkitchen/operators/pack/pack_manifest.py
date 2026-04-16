"""Pack manifest operator: write the CutSet through as-is.

This is the simplest terminal stage. The runner already writes the
stage's output to ``cuts.jsonl.gz``; this operator just passes cuts
through without modification. Useful as a pipeline terminator in
configs where no format conversion is needed.
"""

from __future__ import annotations

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet


class PackManifestConfig(OperatorConfig):
    """pack_manifest has no tunable parameters."""


@register_operator
class PackManifestOperator(Operator):
    """Write a flat manifest (cuts.jsonl.gz) with no audio export."""

    name = "pack_manifest"
    config_cls = PackManifestConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(list(cuts))
