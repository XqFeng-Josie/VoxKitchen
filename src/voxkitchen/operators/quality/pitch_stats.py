"""PitchStats operator: fundamental frequency statistics via PyWorld."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class PitchStatsConfig(OperatorConfig):
    f0_min: float = 50.0  # Hz — ignore F0 below this
    f0_max: float = 2400.0  # Hz — ignore F0 above this
    frame_period_ms: float = 5.0  # analysis hop in ms


@register_operator
class PitchStatsOperator(Operator):
    """Compute pitch (F0) statistics using PyWorld (dio + stonemask).

    More accurate than librosa.pyin for speech. Writes:
      - ``metrics["pitch_mean"]`` — mean F0 in Hz (voiced frames only)
      - ``metrics["pitch_std"]`` — normalized std (0-1 range, pitch-independent)

    A ``pitch_mean`` of 0 means no voiced frames were detected.
    """

    name = "pitch_stats"
    config_cls = PitchStatsConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["pitch"]

    def setup(self) -> None:
        import pyworld  # type: ignore[import-not-found]

        self._pyworld = pyworld

    def process(self, cuts: CutSet) -> CutSet:
        return CutSet(self._process_cut(cut) for cut in cuts)

    def _process_cut(self, cut: Cut) -> Cut:
        assert isinstance(self.config, PitchStatsConfig)
        audio, sr = load_audio_for_cut(cut)
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.astype(np.float64)

        f0, t = self._pyworld.dio(audio, sr, frame_period=self.config.frame_period_ms)
        pitch = self._pyworld.stonemask(audio, f0, t, sr)

        # Filter invalid values
        pitch[pitch < self.config.f0_min] = 0
        pitch[pitch > self.config.f0_max] = 0
        pitch = pitch[pitch > 0]

        if len(pitch) == 0:
            return cut.model_copy(
                update={"metrics": {**cut.metrics, "pitch_mean": 0.0, "pitch_std": 0.0}}
            )

        mean = round(float(np.mean(pitch)), 2)

        # Normalized std: min-max normalize pitch then compute std.
        # This removes the effect of absolute pitch level, making std
        # comparable across speakers with different pitch ranges.
        p_min, p_max = float(np.min(pitch)), float(np.max(pitch))
        if p_max > p_min:
            norm = (pitch - p_min) / (p_max - p_min)
            std = round(float(np.std(norm)), 4)
        else:
            std = 0.0

        return cut.model_copy(
            update={"metrics": {**cut.metrics, "pitch_mean": mean, "pitch_std": std}}
        )
