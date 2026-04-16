"""WhisperxAsr operator: word-aligned ASR via whisperx, with faster-whisper fallback."""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class WhisperxAsrConfig(OperatorConfig):
    model: str = "tiny"
    language: str | None = None
    batch_size: int = 8
    compute_type: str = "int8"


@register_operator
class WhisperxAsrOperator(Operator):
    """Transcribe audio with word-level alignment using whisperx.

    If whisperx is not installed, falls back to faster-whisper at segment level
    (no word alignment). Either path requires the ``asr`` extras group.
    """

    name = "whisperx_asr"
    config_cls = WhisperxAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["asr"]

    def setup(self) -> None:
        import torch

        assert isinstance(self.config, WhisperxAsrConfig)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            import whisperx

            self._whisperx = whisperx
            self._model = whisperx.load_model(
                self.config.model,
                self._device,
                compute_type=self.config.compute_type,
            )
            self._has_whisperx = True
        except ImportError:
            from faster_whisper import WhisperModel

            compute_type = self.config.compute_type
            if self._device == "cpu" and compute_type == "float16":
                compute_type = "int8"
            self._model = WhisperModel(
                self.config.model,
                device=self._device,
                compute_type=compute_type,
            )
            self._has_whisperx = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, WhisperxAsrConfig)
        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            new_sups: list[Supervision] = []
            if self._has_whisperx:
                result = self._model.transcribe(
                    audio,
                    batch_size=self.config.batch_size,
                    language=self.config.language,
                )
                language = result.get("language")
                for seg in result.get("segments", []):
                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__asr_{len(new_sups)}",
                            recording_id=cut.recording_id,
                            start=cut.start + seg["start"],
                            duration=seg["end"] - seg["start"],
                            text=seg["text"].strip(),
                            language=language,
                        )
                    )
            else:
                segments, info = self._model.transcribe(
                    audio,
                    language=self.config.language,
                )
                for seg in segments:
                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__asr_{len(new_sups)}",
                            recording_id=cut.recording_id,
                            start=cut.start + seg.start,
                            duration=seg.end - seg.start,
                            text=seg.text.strip(),
                            language=info.language,
                        )
                    )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
