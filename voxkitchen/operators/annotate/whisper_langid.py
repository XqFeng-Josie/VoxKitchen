"""WhisperLangid operator: language identification using Whisper.

Uses the first 30 seconds of audio to detect language — fast and accurate
for 99 languages. Works with either ``openai-whisper`` or ``faster-whisper``.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language


class WhisperLangidConfig(OperatorConfig):
    model: str = "tiny"  # tiny is sufficient for language detection
    backend: str = "auto"  # "auto", "openai", or "faster-whisper"


@register_operator
class WhisperLangidOperator(Operator):
    """Detect the spoken language of each cut using Whisper.

    Adds a Supervision with the detected ``language`` code (e.g., "en",
    "zh", "ja"). Uses only the first 30 seconds for detection — fast
    even on long recordings.

    Backend selection (``backend`` config):
      - ``auto``: prefer faster-whisper, fall back to openai-whisper
      - ``openai``: use openai-whisper (macOS-safe)
      - ``faster-whisper``: use faster-whisper (faster on GPU)
    """

    name = "whisper_langid"
    config_cls = WhisperLangidConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["whisper"]

    def setup(self) -> None:
        assert isinstance(self.config, WhisperLangidConfig)
        backend = self.config.backend

        if backend == "auto":
            try:
                self._setup_faster_whisper()
                self._backend = "faster-whisper"
                return
            except ImportError:
                pass
            self._setup_openai_whisper()
            self._backend = "openai"
        elif backend == "faster-whisper":
            self._setup_faster_whisper()
            self._backend = "faster-whisper"
        else:
            self._setup_openai_whisper()
            self._backend = "openai"

    def _setup_faster_whisper(self) -> None:
        import torch
        from faster_whisper import WhisperModel

        assert isinstance(self.config, WhisperLangidConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self._fw_model = WhisperModel(self.config.model, device=device, compute_type=compute_type)

    def _setup_openai_whisper(self) -> None:
        import torch
        import whisper  # type: ignore[import-not-found]

        assert isinstance(self.config, WhisperLangidConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ow_model = whisper.load_model(self.config.model, device=device)
        self._ow_device = device

    def process(self, cuts: CutSet) -> CutSet:
        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)

            # Resample to 16kHz if needed
            if sr != 16000:
                from scipy.signal import resample as scipy_resample

                new_len = int(len(audio) * 16000 / sr)
                audio = scipy_resample(audio, new_len).astype(np.float32)

            lang = self._detect(audio)

            sup = Supervision(
                id=f"{cut.id}__{self.ctx.stage_name}",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                language=normalize_language(lang),
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)

    def _detect(self, audio: np.ndarray) -> str:  # type: ignore[type-arg]
        # Use first 30 seconds max
        audio_30s = audio[: 16000 * 30]

        if self._backend == "faster-whisper":
            _segments, info = self._fw_model.transcribe(audio_30s, beam_size=1)
            return info.language or "unknown"

        # openai-whisper — whisper module already imported in setup
        import whisper

        mel = whisper.log_mel_spectrogram(
            whisper.pad_or_trim(audio_30s),
            n_mels=self._ow_model.dims.n_mels,
        ).to(self._ow_device)
        _, probs = self._ow_model.detect_language(mel)
        return str(max(probs, key=probs.get))
