"""SilenceSplit operator: split audio on silence via librosa.effects.split."""

from __future__ import annotations

from typing import ClassVar

import librosa
import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.time import now_utc


class SilenceSplitConfig(OperatorConfig):
    top_db: int = 30  # dB below peak to treat as silence
    min_duration: float = 0.1  # minimum segment duration in seconds


@register_operator
class SilenceSplitOperator(Operator):
    """Split each Cut on silent regions using librosa.effects.split.

    Returns one child Cut per non-silent interval.  No new audio is written.
    """

    name = "silence_split"
    config_cls = SilenceSplitConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["segment"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SilenceSplitConfig)
        out: list[Cut] = []
        for cut in cuts:
            out.extend(self._segment_cut(cut))
        return CutSet(out)

    def _segment_cut(self, cut: Cut) -> list[Cut]:
        assert isinstance(self.config, SilenceSplitConfig)
        audio, sr = load_audio_for_cut(cut)

        # Ensure mono float32
        if audio.ndim == 2:
            audio = audio[:, 0]

        # librosa.effects.split behaves unexpectedly on all-zero audio:
        # power_to_db(ref=max) returns 0 dB when max is 0, treating the whole
        # file as non-silent.  Guard against this explicitly.
        if np.max(np.abs(audio)) < 1e-9:
            return []

        intervals = librosa.effects.split(y=audio, top_db=self.config.top_db)

        generated_by = f"silence_split@top_db{self.config.top_db}"
        stage_name = getattr(getattr(self, "ctx", None), "stage_name", "unknown")
        pipeline_run_id = getattr(getattr(self, "ctx", None), "pipeline_run_id", "unknown")

        children: list[Cut] = []
        for idx, (start_sample, end_sample) in enumerate(intervals):
            start_sec = float(start_sample) / sr
            duration_sec = float(end_sample - start_sample) / sr
            if duration_sec < self.config.min_duration:
                continue
            children.append(
                Cut(
                    id=f"{cut.id}__sil{idx}",
                    recording_id=cut.recording_id,
                    start=cut.start + start_sec,
                    duration=duration_sec,
                    recording=cut.recording,
                    supervisions=[],
                    metrics={},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=generated_by,
                        stage_name=stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return children
