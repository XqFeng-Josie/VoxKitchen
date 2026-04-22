"""WhisperOpenaiAsr operator: ASR using OpenAI's official whisper package.

Unlike faster-whisper (CTranslate2-based), this uses pure PyTorch and
works reliably on macOS ARM64 without deadlock issues.

Install: ``pip install openai-whisper``
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language


class WhisperOpenaiAsrConfig(OperatorConfig):
    model: str = "tiny"  # tiny/base/small/medium/large/turbo
    language: str | None = None  # None = auto-detect
    beam_size: int = 5
    fp16: bool = True  # False for CPU


@register_operator
class WhisperOpenaiAsrOperator(Operator):
    """Transcribe audio using OpenAI's official whisper (pure PyTorch).

    Works on both CPU and GPU. On CPU, set ``fp16: false``. Auto-detects
    CUDA and falls back to CPU transparently.

    This is the recommended ASR operator for macOS where CTranslate2-based
    operators (faster_whisper_asr) may deadlock.
    """

    name = "whisper_openai_asr"
    config_cls = WhisperOpenaiAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["whisper"]

    def setup(self) -> None:
        import torch
        import whisper  # type: ignore[import-not-found]

        assert isinstance(self.config, WhisperOpenaiAsrConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fp16 = self.config.fp16 and device == "cuda"
        self._model = whisper.load_model(self.config.model, device=device)
        self._device = device
        self._fp16 = fp16

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, WhisperOpenaiAsrConfig)
        import numpy as np

        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]
            audio = audio.astype(np.float32)

            # whisper expects 16kHz; resample if needed
            if sr != 16000:
                from scipy.signal import resample as scipy_resample

                new_len = int(len(audio) * 16000 / sr)
                audio = scipy_resample(audio, new_len).astype(np.float32)

            result = self._model.transcribe(
                audio,
                language=self.config.language,
                beam_size=self.config.beam_size,
                fp16=self._fp16,
            )

            new_sups: list[Supervision] = []
            detected_lang = normalize_language(result.get("language") or self.config.language)
            for seg in result.get("segments", []):
                new_sups.append(
                    Supervision(
                        id=f"{cut.id}__{self.ctx.stage_name}_{len(new_sups)}",
                        recording_id=cut.recording_id,
                        start=cut.start + seg["start"],
                        duration=seg["end"] - seg["start"],
                        text=seg["text"].strip(),
                        language=detected_lang,
                    )
                )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
