"""Loudness normalization operator using pyloudnorm (EBU R 128 / LUFS)."""

from __future__ import annotations

from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut, recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

_Audio = np.ndarray[Any, np.dtype[np.float32]]


class LoudnessNormalizeConfig(OperatorConfig):
    target_lufs: float = -23.0


@register_operator
class LoudnessNormalizeOperator(Operator):
    """Normalize audio loudness to a target LUFS level (EBU R 128)."""

    name = "loudness_normalize"
    config_cls = LoudnessNormalizeConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def setup(self) -> None:
        import pyloudnorm

        self._pyloudnorm = pyloudnorm

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, LoudnessNormalizeConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        target_lufs = self.config.target_lufs

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            meter = self._pyloudnorm.Meter(sr)
            loudness = meter.integrated_loudness(audio)
            normalized: _Audio = self._pyloudnorm.normalize.loudness(audio, loudness, target_lufs)
            normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)

            out_path = derived_dir / f"{cut.id}.wav"
            save_audio(out_path, normalized, sr)
            lufs_tag = str(int(abs(target_lufs)))
            new_rec = recording_from_file(
                out_path, recording_id=f"{cut.recording_id}_lufs{lufs_tag}"
            )

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__lufs{lufs_tag}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"loudness_normalize@{target_lufs}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return CutSet(out_cuts)
