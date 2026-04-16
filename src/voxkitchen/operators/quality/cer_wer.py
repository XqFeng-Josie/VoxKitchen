"""CER/WER operator: compute character and word error rates.

Compares ASR hypothesis text (from supervisions) against a reference
text (from cut.custom). Writes ``metrics["cer"]`` and ``metrics["wer"]``.
"""

from __future__ import annotations

import logging

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet

logger = logging.getLogger(__name__)


class CerWerConfig(OperatorConfig):
    hypothesis_field: str = "text"
    reference_field: str = "reference_text"


@register_operator
class CerWerOperator(Operator):
    """Compute CER and WER between ASR output and reference text."""

    name = "cer_wer"
    config_cls = CerWerConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, CerWerConfig)
        out: list[Cut] = []
        for cut in cuts:
            ref = cut.custom.get(self.config.reference_field)
            if ref is None:
                out.append(cut)
                continue

            hyp = ""
            for sup in cut.supervisions:
                if sup.text:
                    hyp = sup.text
                    break

            ref_str = str(ref)
            cer = _edit_distance(list(hyp), list(ref_str)) / max(len(ref_str), 1)
            wer = _edit_distance(hyp.split(), ref_str.split()) / max(len(ref_str.split()), 1)

            metrics = dict(cut.metrics)
            metrics["cer"] = round(min(cer, 1.0), 4)
            metrics["wer"] = round(min(wer, 1.0), 4)
            out.append(cut.model_copy(update={"metrics": metrics}))
        return CutSet(out)


def _edit_distance(hyp: list[str], ref: list[str]) -> int:
    """Levenshtein edit distance between two sequences."""
    n, m = len(hyp), len(ref)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if hyp[i - 1] == ref[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]
