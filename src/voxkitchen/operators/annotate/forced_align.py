"""Forced alignment operator using ctc-forced-aligner.

Aligns existing text (from a prior ASR stage) to audio at word level.
Results are stored in ``cut.custom["word_alignments"]`` as a list of
``{"text": str, "start": float, "end": float}`` dicts.

Cuts without text supervisions are passed through unchanged.
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
    model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner"
    language: str = "eng"  # ISO 639-3 code


@register_operator
class ForcedAlignOperator(Operator):
    name = "forced_align"
    config_cls = ForcedAlignConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras = ["align"]

    _model: Any
    _tokenizer: Any
    _device: str

    def setup(self) -> None:
        import torch
        from ctc_forced_aligner import load_alignment_model

        assert isinstance(self.config, ForcedAlignConfig)
        device = self.ctx.device if hasattr(self.ctx, "device") else "cpu"
        self._device = device
        self._model, self._tokenizer = load_alignment_model(
            device=device,
            dtype=torch.float32,
        )

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
        import torch
        from ctc_forced_aligner import (
            generate_emissions,
            get_alignments,
            get_spans,
            postprocess_results,
        )

        audio, sr = load_audio_for_cut(cut)

        waveform = torch.from_numpy(audio).to(self._device)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        emissions, stride = generate_emissions(self._model, waveform, batch_size=1)

        tokens = self._tokenizer(text, return_tensors="pt")
        token_ids = tokens["input_ids"].squeeze().tolist()

        blank_id = self._tokenizer.pad_token_id or 0
        aligned_tokens, _ = get_alignments(emissions, token_ids, blank_id)
        spans = get_spans(aligned_tokens, blank_id)
        results = postprocess_results(spans, stride, self._tokenizer, waveform)

        word_alignments = []
        for segment in results:
            word_alignments.append(
                {
                    "text": segment.get("text", segment.get("word", "")),
                    "start": round(segment.get("start", 0.0), 3),
                    "end": round(segment.get("end", 0.0), 3),
                }
            )
        return word_alignments

    def teardown(self) -> None:
        self._model = None
        self._tokenizer = None
