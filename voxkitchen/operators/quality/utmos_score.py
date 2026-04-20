"""UTMOS operator: speech naturalness MOS prediction.

Uses the ``speechmos`` package (same as DNSMOS) which provides UTMOS
via a PyTorch model. Good for evaluating TTS/voice conversion output.

Install: ``pip install speechmos``
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut

_UTMOS_SR = 16000


class UtmosScoreConfig(OperatorConfig):
    """No configurable parameters."""


@register_operator
class UtmosScoreOperator(Operator):
    """Predict speech naturalness MOS using UTMOS (no reference needed).

    Writes ``metrics["utmos"]`` — predicted MOS score (1-5).
    Higher is better. Scores > 4.0 indicate natural-sounding speech.

    Useful for filtering synthetic/degraded audio from training data.
    """

    name = "utmos_score"
    config_cls = UtmosScoreConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["dnsmos"]

    def setup(self) -> None:
        from speechmos import utmos  # type: ignore[import-not-found]

        self._utmos = utmos

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        audio, sr = load_audio_for_cut(cut)
        if audio.ndim == 2:
            audio = audio[:, 0]

        if sr != _UTMOS_SR:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * _UTMOS_SR / sr)
            audio = scipy_resample(audio, new_len).astype(np.float32)

        result = self._utmos.run(audio, sr=_UTMOS_SR)
        score = result["mos"] if isinstance(result, dict) else float(result)

        return cut.model_copy(update={"metrics": {**cut.metrics, "utmos": float(round(score, 3))}})
