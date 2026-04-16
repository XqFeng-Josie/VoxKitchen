"""DNSMOS operator: Microsoft Deep Noise Suppression MOS scoring.

Provides both P.835 (sig/bak/ovrl) and P.808 (overall MOS) scores.
Uses the ``speechmos`` package which wraps the official ONNX models.

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

_DNSMOS_SR = 16000


class DnsmosScoreConfig(OperatorConfig):
    use_gpu: bool = False


@register_operator
class DnsmosScoreOperator(Operator):
    """Score audio quality using Microsoft DNSMOS (no reference needed).

    Writes four metrics:
      - ``dnsmos_ovrl`` — P.835 overall quality (1-5)
      - ``dnsmos_sig`` — P.835 speech signal quality (1-5)
      - ``dnsmos_bak`` — P.835 background noise quality (1-5)
      - ``dnsmos_p808`` — P.808 overall MOS (1-5)

    Higher is better. Typically ``dnsmos_ovrl > 3.0`` is considered
    acceptable for training data.
    """

    name = "dnsmos_score"
    config_cls = DnsmosScoreConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["dnsmos"]

    def setup(self) -> None:
        from speechmos import dnsmos  # type: ignore[import-not-found]

        self._dnsmos = dnsmos

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        audio, sr = load_audio_for_cut(cut)
        if audio.ndim == 2:
            audio = audio[:, 0]

        # Clamp to [-1, 1] — wav files decoded as int16 may exceed this range
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # DNSMOS expects 16kHz
        if sr != _DNSMOS_SR:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * _DNSMOS_SR / sr)
            audio = scipy_resample(audio, new_len).astype(np.float32)

        result = self._dnsmos.run(audio, sr=_DNSMOS_SR)

        return cut.model_copy(
            update={
                "metrics": {
                    **cut.metrics,
                    "dnsmos_ovrl": float(round(result["ovrl_mos"], 3)),
                    "dnsmos_sig": float(round(result["sig_mos"], 3)),
                    "dnsmos_bak": float(round(result["bak_mos"], 3)),
                    "dnsmos_p808": float(round(result["p808_mos"], 3)),
                }
            }
        )
