"""Speaker embedding extraction operator.

Extracts a fixed-dimension speaker embedding vector from each audio cut.
Supports WeSpeaker and SpeechBrain backends.

Embeddings are stored in ``cut.custom["speaker_embedding"]`` as a list of
floats. Useful for speaker verification, clustering, and deduplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class SpeakerEmbedConfig(OperatorConfig):
    method: str = "wespeaker"  # "wespeaker" or "speechbrain"
    wespeaker_model: str = "english"
    speechbrain_model: str = "speechbrain/spkrec-ecapa-voxceleb"


@register_operator
class SpeakerEmbedOperator(Operator):
    """Extract speaker embedding vectors using WeSpeaker or SpeechBrain."""

    name = "speaker_embed"
    config_cls = SpeakerEmbedConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["speaker"]

    _model: Any

    def setup(self) -> None:
        assert isinstance(self.config, SpeakerEmbedConfig)
        if self.config.method == "wespeaker":
            self._setup_wespeaker()
        elif self.config.method == "speechbrain":
            self._setup_speechbrain()
        else:
            raise ValueError(
                f"unknown method: {self.config.method!r}, use 'wespeaker' or 'speechbrain'"
            )

    def _setup_wespeaker(self) -> None:
        import wespeaker

        assert isinstance(self.config, SpeakerEmbedConfig)
        self._model = wespeaker.load_model(self.config.wespeaker_model)

    def _setup_speechbrain(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        assert isinstance(self.config, SpeakerEmbedConfig)
        device = self.ctx.device if hasattr(self.ctx, "device") else "cpu"
        self._model = EncoderClassifier.from_hparams(
            source=self.config.speechbrain_model,
            run_opts={"device": device},
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SpeakerEmbedConfig)
        if self.config.method == "wespeaker":
            return self._process_wespeaker(cuts)
        return self._process_speechbrain(cuts)

    def _process_wespeaker(self, cuts: CutSet) -> CutSet:
        import torch

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            # WeSpeaker expects a torch tensor, not numpy array
            pcm = (
                torch.from_numpy(audio).unsqueeze(0) if audio.ndim == 1 else torch.from_numpy(audio)
            )
            embedding = self._model.extract_embedding_from_pcm(pcm, sample_rate=sr)
            if hasattr(embedding, "numpy"):
                embedding = embedding.numpy()
            emb_list = np.array(embedding).flatten().tolist()

            custom = dict(cut.custom) if cut.custom else {}
            custom["speaker_embedding"] = emb_list
            assert isinstance(self.config, SpeakerEmbedConfig)
            custom["speaker_embedding_model"] = f"wespeaker/{self.config.wespeaker_model}"
            custom["speaker_embedding_dim"] = len(emb_list)

            out_cuts.append(cut.model_copy(update={"custom": custom}))
        return CutSet(out_cuts)

    def _process_speechbrain(self, cuts: CutSet) -> CutSet:
        import torch

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            waveform = torch.from_numpy(audio).unsqueeze(0)
            embedding = self._model.encode_batch(waveform).squeeze().detach().cpu().numpy()
            emb_list = embedding.flatten().tolist()

            custom = dict(cut.custom) if cut.custom else {}
            custom["speaker_embedding"] = emb_list
            assert isinstance(self.config, SpeakerEmbedConfig)
            custom["speaker_embedding_model"] = f"speechbrain/{self.config.speechbrain_model}"
            custom["speaker_embedding_dim"] = len(emb_list)

            out_cuts.append(cut.model_copy(update={"custom": custom}))
        return CutSet(out_cuts)

    def teardown(self) -> None:
        self._model = None
