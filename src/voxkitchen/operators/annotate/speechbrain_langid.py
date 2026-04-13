"""SpeechBrainLangId operator: language identification via SpeechBrain."""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class SpeechBrainLangIdConfig(OperatorConfig):
    model: str = "speechbrain/lang-id-voxlingua107-ecapa"


@register_operator
class SpeechBrainLangIdOperator(Operator):
    """Add a language-identification Supervision to each Cut using SpeechBrain.

    Uses the VoxLingua107 ECAPA-TDNN model by default. Runs on CPU with
    automatic fallback from CUDA.
    """

    name = "speechbrain_langid"
    config_cls = SpeechBrainLangIdConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["classify"]

    def setup(self) -> None:
        import torch

        assert isinstance(self.config, SpeechBrainLangIdConfig)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        from speechbrain.inference.classifiers import EncoderClassifier

        self._classifier = EncoderClassifier.from_hparams(
            source=self.config.model,
            run_opts={"device": self._device},
        )

    def process(self, cuts: CutSet) -> CutSet:
        import torch

        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            # Ensure 1-D mono (speechbrain expects (samples,) wrapped in batch dim)
            if audio.ndim == 2:
                audio = audio[:, 0]
            tensor = torch.from_numpy(audio)
            _out_prob, _score, _index, text_lab = self._classifier.classify_batch(
                tensor.unsqueeze(0)
            )
            lang = text_lab[0]
            sup = Supervision(
                id=f"{cut.id}__langid",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                language=lang,
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)
