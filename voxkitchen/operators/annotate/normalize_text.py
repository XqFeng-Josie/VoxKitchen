"""NormalizeText operator: clean ASR transcripts before export.

Removes model markup (e.g. SenseVoice ``<|zh|><|HAPPY|>`` tags), collapses
whitespace (e.g. Paraformer's inter-character spaces in CJK), and optionally
lowercases. CPU-only, no models, no audio.
"""

from __future__ import annotations

import re
from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet

_TAG_RE = re.compile(r"<\|[^|>]*\|>")
_WS_RE = re.compile(r"\s+")
# A space flanked on both sides by non-ASCII (CJK) characters — e.g. Paraformer's
# inter-character spacing. Spaces at CJK<->ASCII or ASCII<->ASCII boundaries stay.
_CJK_SPACE_RE = re.compile(r"(?<=[^\x00-\x7f]) (?=[^\x00-\x7f])")


class NormalizeTextConfig(OperatorConfig):
    strip_tags: bool = True
    collapse_spaces: bool = True
    lowercase: bool = False


def _normalize(text: str, *, strip_tags: bool, collapse_spaces: bool, lowercase: bool) -> str:
    if strip_tags:
        text = _TAG_RE.sub("", text)
    if collapse_spaces:
        text = _WS_RE.sub(" ", text).strip()
        # drop spaces that sit between two non-ASCII (CJK) characters
        text = _CJK_SPACE_RE.sub("", text)
    if lowercase:
        text = text.lower()
    return text


@register_operator
class NormalizeTextOperator(Operator):
    """Normalize ``supervisions[].text`` in place: strip tags, collapse spaces."""

    name = "normalize_text"
    config_cls = NormalizeTextConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False
    reads: ClassVar[list[str]] = ["supervisions.text"]
    writes: ClassVar[list[str]] = ["supervisions.text"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, NormalizeTextConfig)
        out = []
        for cut in cuts:
            new_sups = []
            for s in cut.supervisions:
                if s.text:
                    new_text = _normalize(
                        s.text,
                        strip_tags=self.config.strip_tags,
                        collapse_spaces=self.config.collapse_spaces,
                        lowercase=self.config.lowercase,
                    )
                    new_sups.append(s.model_copy(update={"text": new_text}))
                else:
                    new_sups.append(s)
            out.append(cut.model_copy(update={"supervisions": new_sups}))
        return CutSet(out)
