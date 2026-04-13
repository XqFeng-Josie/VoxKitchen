"""SpeechBrainGender operator: gender classification via SpeechBrain.

NOTE: SpeechBrain does not ship a dedicated gender-classification model in its
standard model zoo. The default model (``speechbrain/spkrec-ecapa-voxceleb``)
is a speaker-recognition model and does **not** reliably classify gender.
This operator provides the correct interface and registration so that a
suitable model can be swapped in via the ``model`` config field.  If the
configured model fails to load, ``setup()`` logs a warning and the operator
becomes a no-op that returns cuts unchanged.
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class SpeechBrainGenderConfig(OperatorConfig):
    model: str = "speechbrain/spkrec-ecapa-voxceleb"  # placeholder default


@register_operator
class SpeechBrainGenderOperator(Operator):
    """Add a gender Supervision to each Cut using SpeechBrain.

    The default model is a placeholder (speaker recognition, not gender
    classification). Override ``model`` in the config to use a true gender
    classifier when one becomes available.  If the model fails to load the
    operator silently becomes a no-op.
    """

    name = "speechbrain_gender"
    config_cls = SpeechBrainGenderConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["classify"]

    def setup(self) -> None:
        import logging

        import torch

        assert isinstance(self.config, SpeechBrainGenderConfig)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            from speechbrain.inference.classifiers import EncoderClassifier

            self._classifier = EncoderClassifier.from_hparams(
                source=self.config.model,
                run_opts={"device": self._device},
            )
            self._available = True
        except Exception:
            logging.getLogger(__name__).warning(
                "speechbrain_gender: model %r not available, operator will be a no-op",
                self.config.model,
            )
            self._available = False

    def process(self, cuts: CutSet) -> CutSet:
        if not self._available:
            return cuts

        import torch

        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]
            tensor = torch.from_numpy(audio)
            _out_prob, _score, _index, text_lab = self._classifier.classify_batch(
                tensor.unsqueeze(0)
            )
            label = text_lab[0]
            sup = Supervision(
                id=f"{cut.id}__gender",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                custom={"gender_label": label},
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)
