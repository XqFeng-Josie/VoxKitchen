"""AudioFingerprintDedup operator: remove near-duplicate audio via MFCC + simhash."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut


class AudioFingerprintDedupConfig(OperatorConfig):
    similarity_threshold: int = 3  # max hamming distance to consider duplicate


@register_operator
class AudioFingerprintDedupOperator(Operator):
    """Remove near-duplicate cuts using MFCC mean features + simhash.

    For each cut, a 13-coefficient MFCC mean vector is computed and hashed
    with simhash.  Cuts whose hash is within ``similarity_threshold`` bits
    (hamming distance) of any previously seen hash are dropped as duplicates.
    """

    name = "audio_fingerprint_dedup"
    config_cls = AudioFingerprintDedupConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["segment", "quality"]

    def setup(self) -> None:
        import librosa  # — validates dependency is installed
        from simhash import Simhash

        self._librosa = librosa
        self._Simhash = Simhash

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, AudioFingerprintDedupConfig)
        threshold = self.config.similarity_threshold
        seen: list[Any] = []
        kept: list[Cut] = []

        for cut in cuts:
            h = self._fingerprint(cut)
            if any(h.distance(prev) <= threshold for prev in seen):
                continue
            seen.append(h)
            kept.append(cut)

        return CutSet(kept)

    def _fingerprint(self, cut: Cut) -> Any:
        audio, sr = load_audio_for_cut(cut)

        # Ensure mono float32
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        mfcc = self._librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean: np.ndarray[tuple[int], np.dtype[np.float32]] = mfcc.mean(axis=1)

        features = [f"{i}:{v:.2f}" for i, v in enumerate(mfcc_mean)]
        return self._Simhash(features)
