"""Speaker similarity operator: cosine similarity against a reference embedding.

Compares each cut's speaker embedding (from a prior ``speaker_embed`` stage)
against a reference embedding loaded from a .npy file. Writes the cosine
similarity score to ``metrics["speaker_similarity"]`` (0-1).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet

logger = logging.getLogger(__name__)


class SpeakerSimilarityConfig(OperatorConfig):
    reference_path: str
    embedding_key: str = "speaker_embedding"


@register_operator
class SpeakerSimilarityOperator(Operator):
    """Score speaker similarity against a reference embedding (cosine)."""

    name = "speaker_similarity"
    config_cls = SpeakerSimilarityConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    _ref: Any

    def setup(self) -> None:
        assert isinstance(self.config, SpeakerSimilarityConfig)
        ref: np.ndarray[Any, Any] = np.load(self.config.reference_path).flatten().astype(np.float32)
        norm = float(np.linalg.norm(ref))
        if norm > 0:
            ref = ref / norm
        self._ref = ref

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SpeakerSimilarityConfig)
        key = self.config.embedding_key
        out: list[Cut] = []
        for cut in cuts:
            emb_raw = cut.custom.get(key)
            if emb_raw is None:
                logger.warning("cut %s has no %s, setting similarity=0", cut.id, key)
                sim = 0.0
            else:
                emb: Any = np.array(emb_raw, dtype=np.float32).flatten()
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    emb = emb / norm
                sim = float(np.dot(self._ref, emb))
                sim = max(0.0, min(1.0, sim))

            metrics = dict(cut.metrics)
            metrics["speaker_similarity"] = round(sim, 4)
            out.append(cut.model_copy(update={"metrics": metrics}))
        return CutSet(out)
