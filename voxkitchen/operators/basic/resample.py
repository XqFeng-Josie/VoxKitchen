"""Resample operator: change sample rate of audio files."""

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


class ResampleConfig(OperatorConfig):
    target_sr: int
    target_channels: int | None = None


@register_operator
class ResampleOperator(Operator):
    """Resample audio to a target sample rate and channel count."""

    name = "resample"
    config_cls = ResampleConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ResampleConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        target_sr = self.config.target_sr

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            if sr != target_sr:
                audio = self._resample(audio, sr, target_sr)

            if self.config.target_channels is not None:
                audio = self._adjust_channels(audio, self.config.target_channels)

            out_path = derived_dir / f"{cut.id}.wav"
            save_audio(out_path, audio, target_sr)
            new_rec = recording_from_file(
                out_path, recording_id=f"{cut.recording_id}_rs{target_sr}"
            )

            custom = dict(cut.custom) if cut.custom else {}
            if cut.start > 0 or "origin_start" not in custom:
                custom.setdefault("origin_start", round(cut.start, 3))
                custom.setdefault("origin_end", round(cut.start + cut.duration, 3))
            out_cuts.append(
                Cut(
                    id=f"{cut.id}__rs{target_sr}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"resample@{target_sr}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _resample(audio: _Audio, orig_sr: int, target_sr: int) -> _Audio:
        try:
            import torch
            import torchaudio

            if audio.ndim == 1:
                tensor = torch.from_numpy(audio).unsqueeze(0)
            else:
                tensor = torch.from_numpy(audio.T)
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled = resampler(tensor)
            arr: _Audio
            if resampled.shape[0] == 1:
                arr = resampled.squeeze(0).numpy()
            else:
                arr = resampled.T.numpy()
            return arr
        except ImportError:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * target_sr / orig_sr)
            result: _Audio = scipy_resample(audio, new_len).astype(np.float32)
            return result

    @staticmethod
    def _adjust_channels(audio: _Audio, target_channels: int) -> _Audio:
        if audio.ndim == 1 and target_channels == 1:
            return audio
        if audio.ndim == 2 and target_channels == 1:
            result: _Audio = audio.mean(axis=1).astype(np.float32)
            return result
        if audio.ndim == 1 and target_channels > 1:
            return np.column_stack([audio] * target_channels)
        return audio
