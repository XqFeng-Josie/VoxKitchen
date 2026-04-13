"""Paraformer ASR operator: fast non-autoregressive ASR via FunASR.

Paraformer (https://github.com/modelscope/FunASR) is a non-autoregressive
end-to-end ASR model from Alibaba DAMO Academy. It is significantly faster
than autoregressive models (like Whisper) at comparable accuracy, especially
for Chinese.

Requires: ``pip install funasr``
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class ParaformerAsrConfig(OperatorConfig):
    model: str = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    language: str = "zh"


@register_operator
class ParaformerAsrOperator(Operator):
    """Transcribe audio using Paraformer (FunASR).

    The default model includes built-in VAD and punctuation restoration,
    making it suitable for long-form audio without pre-segmentation.
    Much faster than Whisper for Chinese.
    """

    name = "paraformer_asr"
    config_cls = ParaformerAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["funasr"]

    def setup(self) -> None:
        import torch
        from funasr import AutoModel

        assert isinstance(self.config, ParaformerAsrConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel(
            model=self.config.model,
            device=device,
            disable_update=True,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ParaformerAsrConfig)
        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            result = self._model.generate(
                input=audio,
                input_len=len(audio),
                batch_size_s=0,
            )

            new_sups: list[Supervision] = []
            if result and len(result) > 0:
                for item in result:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    text = text.strip()
                    if text:
                        # Paraformer with VAD returns timestamps per sentence
                        timestamp = item.get("timestamp", []) if isinstance(item, dict) else []
                        if timestamp and len(timestamp) >= 2:
                            start_ms = timestamp[0][0] if isinstance(timestamp[0], list) else 0
                            end_ms = (
                                timestamp[-1][-1]
                                if isinstance(timestamp[-1], list)
                                else int(cut.duration * 1000)
                            )
                            seg_start = start_ms / 1000.0
                            seg_dur = (end_ms - start_ms) / 1000.0
                        else:
                            seg_start = cut.start
                            seg_dur = cut.duration

                        new_sups.append(
                            Supervision(
                                id=f"{cut.id}__paraformer_{len(new_sups)}",
                                recording_id=cut.recording_id,
                                start=seg_start,
                                duration=seg_dur,
                                text=text,
                                language=self.config.language,
                            )
                        )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
