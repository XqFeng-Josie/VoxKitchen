"""Forced alignment operator using ctc-forced-aligner.

Aligns existing text (from a prior ASR stage) to audio at word level.
Results are stored in ``cut.custom["word_alignments"]`` as a list of
``{"text": str, "start": float, "end": float}`` dicts.

Cuts without text supervisions are passed through unchanged.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut, save_audio

logger = logging.getLogger(__name__)


class ForcedAlignConfig(OperatorConfig):
    model_type: str = "MMS_FA"  # torchaudio pipeline: MMS_FA, WAV2VEC2_ASR_BASE_960H, etc.
    language: str = "eng"  # ISO 639-3 code, used for text normalization


@register_operator
class ForcedAlignOperator(Operator):
    name = "forced_align"
    config_cls = ForcedAlignConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["align"]

    _model: Any
    _tmpdir: str

    def setup(self) -> None:
        self._model = None  # lazy-loaded on first use
        self._tmpdir = tempfile.mkdtemp(prefix="vkit-align-")

    def process(self, cuts: CutSet) -> CutSet:
        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._get_text(cut)
            if text is None:
                out_cuts.append(cut)
                continue
            try:
                alignments = self._align(cut, text)
                custom = dict(cut.custom) if cut.custom else {}
                custom["word_alignments"] = alignments
                custom["forced_align_model"] = self.config.model_type
                out_cuts.append(cut.model_copy(update={"custom": custom}))
            except Exception:
                logger.warning("forced alignment failed for cut %s", cut.id, exc_info=True)
                out_cuts.append(cut)
        return CutSet(out_cuts)

    def _get_text(self, cut: Cut) -> str | None:
        """Extract text from the first supervision that has it."""
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def _align(self, cut: Cut, text: str) -> list[dict[str, Any]]:
        from ctc_forced_aligner import get_word_stamps, text_normalize

        audio, sr = load_audio_for_cut(cut)

        # Normalize text: lowercase, remove punctuation, clean up for alignment
        assert isinstance(self.config, ForcedAlignConfig)
        normalized = text_normalize(text, iso_code=self.config.language)
        if not normalized.strip():
            logger.warning("text is empty after normalization for cut %s", cut.id)
            return []

        # get_word_stamps needs file paths, write temp files
        audio_path = Path(self._tmpdir) / f"{cut.id}.wav"
        save_audio(audio_path, audio, sr)

        text_path = Path(self._tmpdir) / f"{cut.id}.txt"
        text_path.write_text(normalized, encoding="utf-8")

        assert isinstance(self.config, ForcedAlignConfig)
        word_timestamps, self._model, _ = get_word_stamps(
            str(audio_path),
            str(text_path),
            model=self._model,
            model_type=self.config.model_type,
        )

        # Clean up temp files
        audio_path.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)

        word_alignments = []
        for w in word_timestamps:
            word_alignments.append(
                {
                    "text": w.get("word", w.get("text", "")),
                    "start": round(w.get("start", 0.0), 3),
                    "end": round(w.get("end", 0.0), 3),
                }
            )
        return word_alignments

    def teardown(self) -> None:
        self._model = None
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)
