"""Mel spectrogram extraction operator.

Extract mel spectrogram features from audio and store as a numpy file.
Used for TTS training (Tacotron2, VITS, FastSpeech2, etc.) and audio
analysis. Each cut gets a .npy file with the mel spectrogram matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class MelExtractConfig(OperatorConfig):
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = 8000.0
    ref_db: float = 20.0  # reference dB for log scaling
    output_dir: str | None = None  # if None, use stage_dir/mel/


@register_operator
class MelExtractOperator(Operator):
    """Extract mel spectrogram and save as .npy file per cut."""

    name = "mel_extract"
    config_cls = MelExtractConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, MelExtractConfig)
        mel_dir = (
            Path(self.config.output_dir) if self.config.output_dir else self.ctx.stage_dir / "mel"
        )
        mel_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            mel = self._compute_mel(audio, sr)

            mel_path = mel_dir / f"{cut.id}.npy"
            np.save(str(mel_path), mel)

            custom = dict(cut.custom) if cut.custom else {}
            custom["mel_path"] = str(mel_path)
            custom["mel_shape"] = list(mel.shape)  # [n_mels, T]
            custom["mel_config"] = {
                "n_fft": self.config.n_fft,
                "hop_length": self.config.hop_length,
                "n_mels": self.config.n_mels,
                "sr": sr,
            }

            metrics = dict(cut.metrics)
            metrics["mel_frames"] = mel.shape[1]

            out_cuts.append(cut.model_copy(update={"custom": custom, "metrics": metrics}))
        return CutSet(out_cuts)

    def _compute_mel(self, audio: np.ndarray, sr: int) -> Any:  # type: ignore[type-arg]
        assert isinstance(self.config, MelExtractConfig)
        try:
            import torch
            import torchaudio

            wav = torch.from_numpy(audio).float().unsqueeze(0)
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                f_min=self.config.fmin,
                f_max=self.config.fmax,
            )
            mel: Any = transform(wav).squeeze(0)
            # Log scale
            mel = torch.clamp(mel, min=1e-5).log10()
            return mel.numpy().astype(np.float32)
        except ImportError:
            import librosa

            S = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                fmin=self.config.fmin,
                fmax=self.config.fmax,
            )
            return librosa.power_to_db(S, ref=self.config.ref_db).astype(np.float32)
