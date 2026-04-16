"""Emotion recognition operator using emotion2vec (via FunASR).

Recognizes speech emotions from audio and stores the predicted emotion
label and per-class scores in ``cut.custom``.

9 emotion classes: angry, disgusted, fearful, happy, neutral,
other, sad, surprised, unknown.
"""

from __future__ import annotations

from typing import Any

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut

EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]


class EmotionRecognizeConfig(OperatorConfig):
    model: str = "iic/emotion2vec_plus_large"
    granularity: str = "utterance"  # "utterance" or "frame"


@register_operator
class EmotionRecognizeOperator(Operator):
    """Recognize speech emotions using emotion2vec (9 classes: angry, happy, sad, ...)."""

    name = "emotion_recognize"
    config_cls = EmotionRecognizeConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["funasr"]

    _model: Any

    def setup(self) -> None:
        import torch
        from funasr import AutoModel

        assert isinstance(self.config, EmotionRecognizeConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel(
            model=self.config.model,
            device=device,
            disable_update=True,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, EmotionRecognizeConfig)
        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)

            result = self._model.generate(
                input=audio,
                input_len=len(audio),
                granularity=self.config.granularity,
                extract_embedding=False,
            )

            # Parse result
            custom = dict(cut.custom) if cut.custom else {}
            if result and len(result) > 0:
                entry = result[0]
                labels = entry.get("labels", [])
                scores = entry.get("scores", [])

                if labels:
                    label_idx = labels[0] if isinstance(labels[0], int) else 0
                    emotion = (
                        EMOTION_LABELS[label_idx] if label_idx < len(EMOTION_LABELS) else "unknown"
                    )
                    custom["emotion"] = emotion
                    custom["emotion_label_idx"] = label_idx

                if scores:
                    score_list = scores[0] if isinstance(scores[0], list) else scores
                    custom["emotion_scores"] = {
                        EMOTION_LABELS[i]: round(float(s), 4)
                        for i, s in enumerate(score_list)
                        if i < len(EMOTION_LABELS)
                    }

            custom["emotion_model"] = self.config.model
            out_cuts.append(cut.model_copy(update={"custom": custom}))
        return CutSet(out_cuts)

    def teardown(self) -> None:
        self._model = None
