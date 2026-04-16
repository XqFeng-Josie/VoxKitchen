"""Neural audio codec tokenization operator.

Encode audio into discrete token sequences using EnCodec or DAC.
Tokens are stored in ``cut.custom["codec_tokens"]`` as a list of
lists (one per codebook layer).

Used for codec-LM TTS training (VALL-E, CosyVoice, Fish-Speech, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class CodecTokenizeConfig(OperatorConfig):
    backend: str = "encodec"  # "encodec" or "dac"
    bandwidth: float = 6.0  # target bandwidth in kbps (encodec only)
    model: str = "encodec_24khz"  # model variant


@register_operator
class CodecTokenizeOperator(Operator):
    """Encode audio into discrete codec tokens (EnCodec / DAC)."""

    name = "codec_tokenize"
    config_cls = CodecTokenizeConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["codec"]

    _model: Any
    _codec_sr: int

    def setup(self) -> None:
        assert isinstance(self.config, CodecTokenizeConfig)
        if self.config.backend == "encodec":
            self._setup_encodec()
        elif self.config.backend == "dac":
            self._setup_dac()
        else:
            raise ValueError(
                f"unknown codec backend: {self.config.backend!r}, use 'encodec' or 'dac'"
            )

    def _setup_encodec(self) -> None:
        from encodec import EncodecModel

        assert isinstance(self.config, CodecTokenizeConfig)
        if self.config.model == "encodec_48khz":
            self._model = EncodecModel.encodec_model_48khz()
            self._codec_sr = 48000
        else:
            self._model = EncodecModel.encodec_model_24khz()
            self._codec_sr = 24000
        self._model.set_target_bandwidth(self.config.bandwidth)
        self._model.eval()

    def _setup_dac(self) -> None:
        import dac

        self._model = dac.DAC.load(dac.utils.download(model_type="44khz"))
        self._codec_sr = 44100
        self._model.eval()

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, CodecTokenizeConfig)
        if self.config.backend == "encodec":
            return self._process_encodec(cuts)
        return self._process_dac(cuts)

    def _process_encodec(self, cuts: CutSet) -> CutSet:
        import torch

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            audio_resampled = self._resample_if_needed(audio, sr, self._codec_sr)

            with torch.no_grad():
                wav = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
                encoded = self._model.encode(wav)
                codes = torch.cat([frame.codes for frame in encoded], dim=-1)
                tokens: list[list[int]] = codes.squeeze(0).tolist()

            custom = dict(cut.custom) if cut.custom else {}
            custom["codec_tokens"] = tokens
            custom["codec_backend"] = "encodec"
            custom["codec_sr"] = self._codec_sr
            custom["codec_n_codebooks"] = len(tokens)
            out_cuts.append(cut.model_copy(update={"custom": custom}))
        return CutSet(out_cuts)

    def _process_dac(self, cuts: CutSet) -> CutSet:
        import torch

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            audio_resampled = self._resample_if_needed(audio, sr, self._codec_sr)

            with torch.no_grad():
                wav = torch.from_numpy(audio_resampled).float().unsqueeze(0).unsqueeze(0)
                x = self._model.preprocess(wav, self._codec_sr)
                z, codes, latents, _, _ = self._model.encode(x)
                tokens: list[list[int]] = codes.squeeze(0).tolist()

            custom = dict(cut.custom) if cut.custom else {}
            custom["codec_tokens"] = tokens
            custom["codec_backend"] = "dac"
            custom["codec_sr"] = self._codec_sr
            custom["codec_n_codebooks"] = len(tokens)
            out_cuts.append(cut.model_copy(update={"custom": custom}))
        return CutSet(out_cuts)

    @staticmethod
    def _resample_if_needed(audio: np.ndarray, orig_sr: int, target_sr: int) -> Any:  # type: ignore[type-arg]
        if orig_sr == target_sr:
            return audio
        try:
            import torch
            import torchaudio

            tensor = torch.from_numpy(audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled: Any = resampler(tensor).squeeze(0).numpy()
            return resampled
        except ImportError:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * target_sr / orig_sr)
            return np.asarray(scipy_resample(audio, new_len), dtype=np.float32)

    def teardown(self) -> None:
        self._model = None
