"""SnrEstimate operator: add peak-to-RMS SNR estimate to cut metrics."""

from __future__ import annotations

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class SnrEstimateConfig(OperatorConfig):
    """No configurable parameters."""


@register_operator
class SnrEstimateOperator(Operator):
    """Estimate SNR via a peak-to-RMS ratio and store it in cut.metrics["snr"].

    This is a rough proxy (not WADA-SNR or model-based) sufficient for v0.1.
    No audio is written; only the metrics dict is updated.
    """

    name = "snr_estimate"
    config_cls = SnrEstimateConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        audio, _sr = load_audio_for_cut(cut)

        # Ensure mono
        if audio.ndim == 2:
            audio = audio[:, 0]

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-10:
            snr_db = 0.0
        else:
            peak = float(np.max(np.abs(audio)))
            snr_db = 20.0 * np.log10(peak / rms)

        return cut.model_copy(update={"metrics": {**cut.metrics, "snr": round(snr_db, 2)}})
