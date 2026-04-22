"""FasterWhisperAsr operator: GPU-capable ASR transcription via faster-whisper."""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language


class FasterWhisperAsrConfig(OperatorConfig):
    model: str = "tiny"
    language: str | None = None  # None = auto-detect
    beam_size: int = 5
    compute_type: str = "int8"  # int8 for CPU, float16 for GPU
    cpu_threads: int = 4  # CTranslate2 thread count (set to 1 if you hit deadlocks on macOS)


@register_operator
class FasterWhisperAsrOperator(Operator):
    """Transcribe audio using faster-whisper and add Supervisions with text + language.

    Uses CTranslate2 for inference. Has GPU and CPU, but on CPU the
    compute_type is coerced to "int8" because float16 is not supported.

    .. warning::
       CTranslate2 may deadlock on macOS ARM64. Use ``whisper_openai_asr``
       on macOS instead.
    """

    name = "faster_whisper_asr"
    config_cls = FasterWhisperAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["asr"]

    def setup(self) -> None:
        import torch
        from faster_whisper import WhisperModel

        assert isinstance(self.config, FasterWhisperAsrConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = self.config.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"  # float16 not supported on CPU
        self._model = WhisperModel(
            self.config.model,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.config.cpu_threads,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, FasterWhisperAsrConfig)
        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            # Ensure 1-D (mono) — faster-whisper expects (samples,)
            if audio.ndim == 2:
                audio = audio[:, 0]
            segments, info = self._model.transcribe(
                audio,
                beam_size=self.config.beam_size,
                language=self.config.language,
            )
            new_sups: list[Supervision] = []
            for seg in segments:  # generator — iterate once
                new_sups.append(
                    Supervision(
                        id=f"{cut.id}__{self.ctx.stage_name}_{len(new_sups)}",
                        recording_id=cut.recording_id,
                        start=cut.start + seg.start,
                        duration=seg.end - seg.start,
                        text=seg.text.strip(),
                        language=normalize_language(info.language),
                    )
                )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
