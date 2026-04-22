"""Qwen3 ASR operator: state-of-the-art multilingual speech recognition.

Qwen3-ASR supports 30 languages + 22 Chinese dialects with excellent
accuracy. Optionally integrates Qwen3-ForcedAligner for word-level
timestamps in the same pass.

Requires: ``pip install qwen-asr``
"""

from __future__ import annotations

from typing import Any, ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language


def _to_qwen3_language(lang: str | None) -> str | None:
    """Convert a language input to the capitalized full name Qwen3-ASR expects.

    Qwen3-ASR requires full English names (``"Chinese"``, ``"English"``, …).
    Delegates normalisation to :func:`normalize_language`, then capitalizes.
    """
    canonical = normalize_language(lang)
    return canonical.capitalize() if canonical else None


class Qwen3AsrConfig(OperatorConfig):
    model: str = "Qwen/Qwen3-ASR-0.6B"  # or Qwen/Qwen3-ASR-1.7B
    language: str | None = None  # None = auto-detect; any code/name accepted
    return_timestamps: bool = False  # word-level timestamps via ForcedAligner
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    max_new_tokens: int = 512


@register_operator
class Qwen3AsrOperator(Operator):
    """Transcribe audio using Qwen3-ASR.

    30 languages + 22 Chinese dialects. Set ``return_timestamps=True``
    to also get word-level timestamps (uses ForcedAligner internally).
    """

    name = "qwen3_asr"
    config_cls = Qwen3AsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["align"]

    _model: Any

    def setup(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        assert isinstance(self.config, Qwen3AsrConfig)
        device = self.ctx.device if hasattr(self.ctx, "device") else "cpu"
        dtype = torch.bfloat16 if "cuda" in device else torch.float32

        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_new_tokens": self.config.max_new_tokens,
        }

        if self.config.return_timestamps:
            kwargs["forced_aligner"] = self.config.aligner_model
            kwargs["forced_aligner_kwargs"] = {
                "dtype": dtype,
                "device_map": device,
            }

        self._model = Qwen3ASRModel.from_pretrained(self.config.model, **kwargs)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, Qwen3AsrConfig)
        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            results = self._model.transcribe(
                audio=(audio, sr),
                language=_to_qwen3_language(self.config.language),
                return_time_stamps=self.config.return_timestamps,
            )

            new_sups: list[Supervision] = []
            if results and len(results) > 0:
                result = results[0]
                text = result.text.strip() if result.text else ""
                # Normalize detected language before storage
                language = normalize_language(result.language or self.config.language)

                if text:
                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__{self.ctx.stage_name}_0",
                            recording_id=cut.recording_id,
                            start=cut.start,
                            duration=cut.duration,
                            text=text,
                            language=language,
                        )
                    )

                custom = dict(cut.custom) if cut.custom else {}
                if (
                    self.config.return_timestamps
                    and hasattr(result, "time_stamps")
                    and result.time_stamps
                ):
                    custom["word_alignments"] = [
                        {
                            "text": seg.text,
                            "start": round(seg.start_time, 3),
                            "end": round(seg.end_time, 3),
                        }
                        for seg in result.time_stamps
                    ]

                updated = cut.model_copy(
                    update={
                        "supervisions": [*cut.supervisions, *new_sups],
                        "custom": custom,
                    }
                )
                out.append(updated)
            else:
                out.append(cut)
        return CutSet(out)

    def teardown(self) -> None:
        self._model = None
