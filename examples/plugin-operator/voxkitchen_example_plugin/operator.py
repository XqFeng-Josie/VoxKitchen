"""Example third-party VoxKitchen operator, registered via an entry point.

A plugin operator is registered through the ``voxkitchen.operators`` entry
point declared in pyproject.toml — NOT via the ``@register_operator`` decorator
(the entry point is the registration mechanism; decorating too would
double-register). Subclass ``Operator``, declare the field contract, and ship.
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.schema.cutset import CutSet


class WordCountConfig(OperatorConfig):
    """No parameters."""


class WordCountOperator(Operator):
    """Count words in each cut's first transcribed supervision.

    Writes the count to ``metrics.word_count``. CPU-only, no audio, no deps —
    a minimal template for a third-party operator.
    """

    name = "word_count"
    config_cls = WordCountConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False
    reads: ClassVar[list[str]] = ["supervisions.text"]
    writes: ClassVar[list[str]] = ["metrics.word_count"]

    def process(self, cuts: CutSet) -> CutSet:
        out = []
        for cut in cuts:
            text = next((s.text for s in cut.supervisions if s.text), "")
            metrics = {**cut.metrics, "word_count": float(len(text.split()))}
            out.append(cut.model_copy(update={"metrics": metrics}))
        return CutSet(out)
