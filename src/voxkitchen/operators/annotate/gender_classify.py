"""Gender classification operator: multiple methods for different accuracy/speed tradeoffs.

Supported methods:

- ``f0`` — Pitch-based (librosa pyin). Fastest, no model download, ~80-85% on clean adult speech.
- ``speechbrain`` — SpeechBrain classifier. Needs a pretrained model, ~95%+ accuracy.
- ``inaspeechsegmenter`` — INA Speech Segmenter. Well-tested, ~90-95%, also detects speech/music.

Users choose the method that fits their accuracy/speed/dependency requirements.

Example YAML::

    - name: gender
      op: gender_classify
      args:
        method: f0              # fast, no model needed
        # method: speechbrain   # more accurate, needs model download
        # method: inaspeechsegmenter  # well-tested, needs tensorflow
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class GenderClassifyConfig(OperatorConfig):
    method: str = "f0"
    # F0 settings
    f0_threshold: float = 165.0  # Hz boundary between male/female
    # SpeechBrain settings
    speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    # inaSpeechSegmenter has no tunable params


@register_operator
class GenderClassifyOperator(Operator):
    """Classify speaker gender using one of several methods.

    Methods:

    - ``f0``: Extract fundamental frequency via librosa's pyin. Male if
      median F0 < threshold (default 165 Hz), else female. Fast, no
      model download, but only ~80-85% accurate on clean adult speech.
      Fails on children, elderly, or noisy audio.

    - ``speechbrain``: Use a SpeechBrain EncoderClassifier. More accurate
      (~95%+) but requires model download. Default model is a speaker
      recognition model (placeholder) — override ``speechbrain_model``
      with a true gender classifier for best results.

    - ``inaspeechsegmenter``: Use INA's speech segmenter which jointly
      detects speech/music/noise and classifies gender. Well-tested in
      broadcast media analysis (~90-95%). Requires ``pip install
      inaSpeechSegmenter`` (uses TensorFlow).
    """

    name = "gender_classify"
    config_cls = GenderClassifyConfig
    device = "cpu"  # all methods work on CPU
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = []

    def setup(self) -> None:
        assert isinstance(self.config, GenderClassifyConfig)
        method = self.config.method

        if method == "f0":
            # No setup needed — librosa is already a dep
            pass
        elif method == "speechbrain":
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sb_classifier = EncoderClassifier.from_hparams(
                source=self.config.speechbrain_model,
                run_opts={"device": device},
            )
        elif method == "inaspeechsegmenter":
            from inaSpeechSegmenter import Segmenter

            self._ina_segmenter = Segmenter()
        else:
            raise ValueError(
                f"unknown gender method: {self.config.method!r}. "
                f"Options: 'f0', 'speechbrain', 'inaspeechsegmenter'"
            )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, GenderClassifyConfig)
        method = self.config.method

        if method == "f0":
            return self._process_f0(cuts)
        elif method == "speechbrain":
            return self._process_speechbrain(cuts)
        elif method == "inaspeechsegmenter":
            return self._process_ina(cuts)
        return cuts

    def _process_f0(self, cuts: CutSet) -> CutSet:
        """Gender via fundamental frequency estimation."""
        import librosa

        assert isinstance(self.config, GenderClassifyConfig)
        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            # pyin returns (f0, voiced_flag, voiced_probs)
            f0, voiced, _ = librosa.pyin(
                audio,
                fmin=50,
                fmax=400,
                sr=sr,
                frame_length=2048,
            )
            voiced_f0 = f0[voiced] if voiced is not None else f0[~np.isnan(f0)]

            if len(voiced_f0) == 0:
                gender: Literal["m", "f", "o"] = "o"  # unknown — no voiced frames
            else:
                median_f0 = float(np.median(voiced_f0))
                gender = "m" if median_f0 < self.config.f0_threshold else "f"

            sup = Supervision(
                id=f"{cut.id}__gender",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                gender=gender,
                custom={
                    "gender_method": "f0",
                    "median_f0": float(np.median(voiced_f0)) if len(voiced_f0) > 0 else None,
                },
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)

    def _process_speechbrain(self, cuts: CutSet) -> CutSet:
        """Gender via SpeechBrain classifier."""
        import torch

        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]
            tensor = torch.from_numpy(audio).unsqueeze(0)
            _prob, _score, _idx, text_lab = self._sb_classifier.classify_batch(tensor)
            label = str(text_lab[0]).lower()

            # Map common labels to m/f/o
            if "male" in label and "female" not in label:
                gender: Literal["m", "f", "o"] = "m"
            elif "female" in label:
                gender = "f"
            else:
                gender = "o"

            sup = Supervision(
                id=f"{cut.id}__gender",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                gender=gender,
                custom={"gender_method": "speechbrain", "raw_label": label},
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)

    def _process_ina(self, cuts: CutSet) -> CutSet:
        """Gender via inaSpeechSegmenter."""
        import tempfile

        import soundfile as sf

        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            # inaSpeechSegmenter needs a file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, sr)
                segments = self._ina_segmenter(f.name)

            # segments is list of (label, start, end) where label in
            # {"male", "female", "noEnergy", "noise", "music"}
            male_dur = sum(e - s for lbl, s, e in segments if lbl == "male")
            female_dur = sum(e - s for lbl, s, e in segments if lbl == "female")

            if male_dur == 0 and female_dur == 0:
                gender: Literal["m", "f", "o"] = "o"
            elif male_dur >= female_dur:
                gender = "m"
            else:
                gender = "f"

            sup = Supervision(
                id=f"{cut.id}__gender",
                recording_id=cut.recording_id,
                start=cut.start,
                duration=cut.duration,
                gender=gender,
                custom={
                    "gender_method": "inaspeechsegmenter",
                    "male_duration": round(male_dur, 2),
                    "female_duration": round(female_dur, 2),
                },
            )
            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, sup]})
            out.append(updated)
        return CutSet(out)
