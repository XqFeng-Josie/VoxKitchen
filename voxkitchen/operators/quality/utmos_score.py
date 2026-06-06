"""UTMOS operator: speech naturalness MOS prediction.

UTMOS22 MOS predictor loaded via ``torch.hub`` from the SpeechMOS repo
(``tarepan/SpeechMOS:v1.2.0``).  No-reference MOS estimate on a 1-5 scale;
higher is better.  Needs only ``torch``, which is present in every VoxKitchen
image including ``slim``.

Note: the previous implementation did ``from speechmos import utmos``, but
``speechmos`` never shipped a ``utmos`` module (it provides aecmos/dnsmos/plcmos
only, across all PyPI versions).  That import failed unconditionally.

Runtime: ``vkit docker run --tag slim <yaml>``
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
    """Predict speech naturalness MOS using UTMOS22 (no reference needed).

    Writes ``metrics["utmos"]`` -- predicted MOS score (1-5).
    Higher is better.  Scores > 4.0 indicate natural-sounding speech.

    The model is loaded via ``torch.hub`` from ``tarepan/SpeechMOS:v1.2.0``
    (pinned tag for reproducibility).  First run downloads ~390 MB of model
    weights into the torch hub cache; subsequent runs are local.

    Useful for filtering synthetic/degraded audio from training data.
    """

    name = "utmos_score"
    config_cls = UtmosScoreConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = []
    reads: ClassVar[list[str]] = ["audio"]
    writes: ClassVar[list[str]] = ["metrics.utmos"]

    def setup(self) -> None:
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # UTMOS22 MOS predictor via torch.hub (SpeechMOS repo). Pinned tag for
        # reproducibility. Needs only torch — the old `from speechmos import utmos`
        # never worked (speechmos ships aecmos/dnsmos/plcmos, never utmos).
        self._predictor = torch.hub.load(  # type: ignore[no-untyped-call]
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).to(self._device)
        self._predictor.eval()

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        import torch

        audio, sr = load_audio_for_cut(cut)
        if audio.ndim == 2:
            audio = audio[:, 0]

        if sr != _UTMOS_SR:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * _UTMOS_SR / sr)
            audio = scipy_resample(audio, new_len).astype(np.float32)

        wav = torch.from_numpy(audio.astype("float32")).unsqueeze(0).to(self._device)
        with torch.no_grad():
            score = float(self._predictor(wav, _UTMOS_SR))

        return cut.model_copy(update={"metrics": {**cut.metrics, "utmos": round(score, 3)}})
