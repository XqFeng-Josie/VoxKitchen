"""SenseVoice ASR operator: multi-language speech understanding via FunASR.

SenseVoice (https://github.com/FunAudioLLM/SenseVoice) is a multi-language
speech understanding model from Alibaba that supports ASR, language
identification, emotion recognition, and audio event detection in a single
model. This operator uses it for ASR transcription.

Requires: ``pip install funasr``
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class SenseVoiceAsrConfig(OperatorConfig):
    model: str = "iic/SenseVoiceSmall"
    language: str = "auto"  # "zh", "en", "ja", "ko", "yue", or "auto"


@register_operator
class SenseVoiceAsrOperator(Operator):
    """Transcribe audio using SenseVoice (FunASR).

    SenseVoice supports Chinese, English, Japanese, Korean, and Cantonese.
    The ``SenseVoiceSmall`` model is fast and accurate for these languages.
    """

    name = "sensevoice_asr"
    config_cls = SenseVoiceAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["funasr"]

    def setup(self) -> None:
        import torch
        from funasr import AutoModel

        assert isinstance(self.config, SenseVoiceAsrConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel(
            model=self.config.model,
            device=device,
            disable_update=True,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SenseVoiceAsrConfig)
        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            result = self._model.generate(
                input=audio,
                input_len=len(audio),
                language=self.config.language,
                use_itn=True,
                batch_size_s=0,
            )

            new_sups: list[Supervision] = []
            if result and len(result) > 0:
                for item in result:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    text = text.strip()
                    if text:
                        new_sups.append(
                            Supervision(
                                id=f"{cut.id}__sensevoice_{len(new_sups)}",
                                recording_id=cut.recording_id,
                                start=cut.start,
                                duration=cut.duration,
                                text=text,
                                language=self.config.language
                                if self.config.language != "auto"
                                else None,
                            )
                        )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
