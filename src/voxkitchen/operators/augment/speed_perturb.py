"""Speed perturbation operator: change playback speed (and pitch) of audio.

The standard speech augmentation technique used in Kaldi, ESPnet, NeMo.
A speed factor of 0.9 makes audio slower/lower-pitched/longer; 1.1 makes
it faster/higher-pitched/shorter.

Implementation: interpret audio as having sample rate ``sr * factor``,
then resample back to ``sr``. This is equivalent to ``sox speed factor``.
"""

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


class SpeedPerturbConfig(OperatorConfig):
    """Config for speed perturbation.

    ``factors`` -- list of speed factors to apply. Each input cut produces
    one output cut per factor. Factor 1.0 copies the original unchanged.
    """

    factors: list[float] = [0.9, 1.0, 1.1]


@register_operator
class SpeedPerturbOperator(Operator):
    """Apply speed perturbation (tempo + pitch change) via resampling."""

    name = "speed_perturb"
    config_cls = SpeedPerturbConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
    required_extras = ["audio"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SpeedPerturbConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            for factor in self.config.factors:
                perturbed = self._speed(audio, sr, factor)
                tag = f"sp{factor:.2f}".replace(".", "")
                out_path = derived_dir / f"{cut.id}__{tag}.wav"
                save_audio(out_path, perturbed, sr)
                new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_{tag}")
                out_cuts.append(
                    Cut(
                        id=f"{cut.id}__{tag}",
                        recording_id=new_rec.id,
                        start=0.0,
                        duration=new_rec.duration,
                        recording=new_rec,
                        supervisions=cut.supervisions,
                        metrics=cut.metrics,
                        provenance=Provenance(
                            source_cut_id=cut.id,
                            generated_by=f"speed_perturb@{factor}",
                            stage_name=self.ctx.stage_name,
                            created_at=now_utc(),
                            pipeline_run_id=self.ctx.pipeline_run_id,
                        ),
                        custom=cut.custom,
                    )
                )
        return CutSet(out_cuts)

    @staticmethod
    def _speed(audio: _Audio, sr: int, factor: float) -> _Audio:
        if factor == 1.0:
            return audio.copy()
        try:
            import torch
            import torchaudio

            if audio.ndim == 1:
                tensor = torch.from_numpy(audio).unsqueeze(0)
            else:
                tensor = torch.from_numpy(audio.T)
            src_sr = int(sr * factor)
            resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=sr)
            result = resampler(tensor)
            if result.shape[0] == 1:
                return result.squeeze(0).numpy().astype(np.float32)
            return result.T.numpy().astype(np.float32)
        except ImportError:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) / factor)
            return scipy_resample(audio, new_len).astype(np.float32)
