"""Forced alignment operator using Qwen3-ForcedAligner.

Aligns existing text (from a prior ASR stage) to audio at word/character
level. Results are stored in ``cut.custom["word_alignments"]`` as a list
of ``{"text": str, "start": float, "end": float}`` dicts.

Cuts without text supervisions are passed through unchanged.

Supports 11 languages: Chinese, English, Cantonese, French, German,
Italian, Japanese, Korean, Portuguese, Russian, Spanish.
"""

from __future__ import annotations

import logging
from typing import Any

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import load_audio_for_cut

logger = logging.getLogger(__name__)


class ForcedAlignConfig(OperatorConfig):
    model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    language: str = "Chinese"  # Chinese, English, French, German, etc.


@register_operator
class ForcedAlignOperator(Operator):
    """Align text to audio at word level using Qwen3-ForcedAligner (11 languages)."""

    name = "forced_align"
    config_cls = ForcedAlignConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["align"]

    _aligner: Any

    def setup(self) -> None:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        assert isinstance(self.config, ForcedAlignConfig)
        device = self.ctx.device if hasattr(self.ctx, "device") else "cpu"
        # Use float32 on CPU, bfloat16 on GPU
        dtype = torch.bfloat16 if "cuda" in device else torch.float32
        self._aligner = Qwen3ForcedAligner.from_pretrained(
            self.config.model,
            dtype=dtype,
            device_map=device,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ForcedAlignConfig)
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
                custom["forced_align_model"] = self.config.model
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
        audio, sr = load_audio_for_cut(cut)

        results = self._aligner.align(
            audio=(audio, sr),
            text=text,
            language=self.config.language,
        )

        word_alignments = []
        if results and len(results) > 0:
            for segment in results[0]:
                word_alignments.append(
                    {
                        "text": segment.text,
                        "start": round(segment.start_time, 3),
                        "end": round(segment.end_time, 3),
                    }
                )
        return word_alignments

    def teardown(self) -> None:
        self._aligner = None
