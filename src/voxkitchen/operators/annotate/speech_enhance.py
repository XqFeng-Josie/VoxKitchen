"""Speech enhancement operator via DeepFilterNet.

Reduces background noise from audio using a neural network model.
DeepFilterNet operates at 48kHz; the operator handles resampling
to/from 48kHz internally so input can be at any sample rate.
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

DF_SAMPLE_RATE = 48000


class SpeechEnhanceConfig(OperatorConfig):
    method: str = "deepfilternet"
    aggressiveness: float = 0.5  # 0.0 to 1.0


@register_operator
class SpeechEnhanceOperator(Operator):
    """Remove background noise using DeepFilterNet neural denoiser."""

    name = "speech_enhance"
    config_cls = SpeechEnhanceConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
    required_extras = ["enhance"]

    _model: Any
    _df_state: Any

    def setup(self) -> None:
        from df.enhance import init_df

        self._model, self._df_state, _ = init_df()

    def process(self, cuts: CutSet) -> CutSet:
        import torch
        from df.enhance import enhance

        assert isinstance(self.config, SpeechEnhanceConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            # Resample to 48kHz if needed
            if sr != DF_SAMPLE_RATE:
                audio_48k = self._resample(audio, sr, DF_SAMPLE_RATE)
            else:
                audio_48k = audio

            # Enhance
            audio_tensor = torch.from_numpy(audio_48k).unsqueeze(0)
            enhanced = enhance(
                self._model,
                self._df_state,
                audio_tensor,
                atten_lim_db=self.config.aggressiveness * 100,
            )
            enhanced_np = enhanced.squeeze().numpy().astype(np.float32)

            # Resample back to original SR
            if sr != DF_SAMPLE_RATE:
                enhanced_np = self._resample(enhanced_np, DF_SAMPLE_RATE, sr)

            enhanced_np = np.clip(enhanced_np, -1.0, 1.0).astype(np.float32)

            out_path = derived_dir / f"{cut.id}__enhanced.wav"
            save_audio(out_path, enhanced_np, sr)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_enhanced")

            custom = dict(cut.custom) if cut.custom else {}
            custom["speech_enhance_method"] = "deepfilternet"

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__enhanced",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="speech_enhance@deepfilternet",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _resample(
        audio: np.ndarray,  # type: ignore[type-arg]
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:  # type: ignore[type-arg]
        try:
            import torch
            import torchaudio

            tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            out: np.ndarray[Any, Any] = resampler(tensor).squeeze(0).numpy().astype(np.float32)
            return out
        except ImportError:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * target_sr / orig_sr)
            out = np.asarray(scipy_resample(audio, new_len), dtype=np.float32)
            return out

    def teardown(self) -> None:
        self._model = None
        self._df_state = None
