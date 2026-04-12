"""Channel merge operator: convert audio to a target channel count."""

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


class ChannelMergeConfig(OperatorConfig):
    target_channels: int = 1


@register_operator
class ChannelMergeOperator(Operator):
    name = "channel_merge"
    config_cls = ChannelMergeConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ChannelMergeConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        target_channels = self.config.target_channels

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            audio = self._adjust_channels(audio, target_channels)

            out_path = derived_dir / f"{cut.id}.wav"
            save_audio(out_path, audio, sr)
            new_rec = recording_from_file(
                out_path, recording_id=f"{cut.recording_id}_ch{target_channels}"
            )

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__ch{target_channels}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"channel_merge@{target_channels}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _adjust_channels(audio: _Audio, target_channels: int) -> _Audio:
        # Already mono (1-D)
        if audio.ndim == 1 and target_channels == 1:
            return audio
        # Stereo/multi → mono: average across channels
        if audio.ndim == 2 and target_channels == 1:
            result: _Audio = audio.mean(axis=1).astype(np.float32)
            return result
        # Mono → multi: duplicate the single channel
        if audio.ndim == 1 and target_channels > 1:
            return np.column_stack([audio] * target_channels)
        # Already the right shape (or ndim==2 and target matches)
        return audio
