"""ClippingDetect operator: measure the ratio of clipped (distorted) samples."""

from __future__ import annotations

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class ClippingDetectConfig(OperatorConfig):
    ceiling: float = 0.99  # absolute amplitude threshold for clipping


@register_operator
class ClippingDetectOperator(Operator):
    """Detect audio clipping and store the ratio of clipped samples.

    Clipping occurs when recording levels are too high, causing the
    waveform to be truncated at the maximum amplitude ceiling. This
    produces harsh distortion that degrades ASR and TTS training.

    Writes ``metrics["clipping_ratio"]`` — fraction of samples whose
    absolute value exceeds ``ceiling`` (default 0.99). A ratio > 0.01
    indicates significant clipping.
    """

    name = "clipping_detect"
    config_cls = ClippingDetectConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ClippingDetectConfig)
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        assert isinstance(self.config, ClippingDetectConfig)
        audio, _sr = load_audio_for_cut(cut)
        if audio.ndim == 2:
            audio = audio[:, 0]

        n_clipped = int(np.sum(np.abs(audio) >= self.config.ceiling))
        ratio = round(n_clipped / max(len(audio), 1), 4)

        return cut.model_copy(update={"metrics": {**cut.metrics, "clipping_ratio": ratio}})
