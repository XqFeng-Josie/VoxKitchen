"""PyannoteDiarize operator: speaker diarization via pyannote.audio."""

from __future__ import annotations

import os
from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class PyannoteDiarizeConfig(OperatorConfig):
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token: str | None = None  # or reads from HF_TOKEN env var


@register_operator
class PyannoteDiarizeOperator(Operator):
    """Add speaker-label Supervisions to each Cut using pyannote.audio.

    Requires accepting the pyannote model user agreement on HuggingFace and
    setting ``HF_TOKEN`` (or passing ``hf_token`` in the config).
    """

    name = "pyannote_diarize"
    config_cls = PyannoteDiarizeConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["diarize"]

    def setup(self) -> None:
        import torch
        from pyannote.audio import Pipeline

        assert isinstance(self.config, PyannoteDiarizeConfig)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = self.config.hf_token or os.environ.get("HF_TOKEN")
        self._pipeline = Pipeline.from_pretrained(self.config.model, use_auth_token=token)
        self._pipeline.to(self._device)

    def process(self, cuts: CutSet) -> CutSet:
        import torch

        assert isinstance(self.config, PyannoteDiarizeConfig)
        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            # pyannote expects (channels, samples) as a float32 tensor
            if audio.ndim == 1:
                waveform = torch.from_numpy(audio).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio.T)
            audio_dict = {"waveform": waveform, "sample_rate": sr}

            diarization_kwargs: dict[str, int] = {}
            if self.config.min_speakers is not None:
                diarization_kwargs["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers is not None:
                diarization_kwargs["max_speakers"] = self.config.max_speakers

            diarization = self._pipeline(audio_dict, **diarization_kwargs)

            new_sups: list[Supervision] = []
            for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                new_sups.append(
                    Supervision(
                        id=f"{cut.id}__diar_{len(new_sups)}",
                        recording_id=cut.recording_id,
                        start=cut.start + turn.start,
                        duration=turn.end - turn.start,
                        speaker=speaker_label,
                    )
                )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
