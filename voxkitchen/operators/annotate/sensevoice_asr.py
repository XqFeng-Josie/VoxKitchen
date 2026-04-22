"""SenseVoice ASR operator: multi-language speech understanding via FunASR.

SenseVoice (https://github.com/FunAudioLLM/SenseVoice) is a multi-language
speech understanding model from Alibaba that supports ASR, language
identification, emotion recognition, and audio event detection in a single
model. This operator uses it for ASR transcription.

Beyond plain ASR, SenseVoice detects per-utterance emotion and audio event
type in the same forward pass. These are stored in ``supervision.custom``:

- ``supervision.custom["emotion"]``: lowercase emotion label
  (``happy`` / ``sad`` / ``angry`` / ``neutral`` / ``disgusted`` /
  ``fearful`` / ``surprised`` / ``unknown``)
- ``supervision.custom["audio_event"]``: detected sound class
  (``Speech`` / ``BGM`` / ``noise``)

``supervision.language`` contains the model-detected language code (e.g.
``"zh"``) rather than the requested language, so auto-detection results are
faithfully recorded even when ``language="auto"``.

Requires: ``pip install funasr``
"""

from __future__ import annotations

import re
from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language as _normalize_language

_SENSEVOICE_TAG_RE = re.compile(r"<\|([^|]*)\|>")

# Maps SenseVoice emotion tag values to lowercase canonical labels.
_EMOTION_MAP: dict[str, str] = {
    "HAPPY": "happy",
    "SAD": "sad",
    "ANGRY": "angry",
    "NEUTRAL": "neutral",
    "DISGUSTED": "disgusted",
    "FEARFUL": "fearful",
    "SURPRISED": "surprised",
    "UNKNOWN": "unknown",
    "EMO_UNKNOWN": "unknown",
}

# Known audio-event tags emitted by SenseVoice.
_AUDIO_EVENTS = {"Speech", "BGM", "noise"}


def _parse_sensevoice_output(raw_text: str) -> tuple[str, str | None, str | None, str | None]:
    """Parse a raw SenseVoice output string into (clean_text, language, emotion, audio_event).

    SenseVoice prepends up to four tags before the transcript::

        <|zh|><|HAPPY|><|Speech|><|withitn|>参观海洋馆。

    Tags are order-independent and optional. Falls back to
    ``funasr.utils.postprocess_utils.rich_transcription_postprocess`` for the
    text cleaning step when FunASR is available.

    Returns a 4-tuple:
        clean_text  - transcript with all tags removed
        language    - canonical language name (e.g. ``"chinese"``), or None
        emotion     - lowercase label (e.g. ``"happy"``), or None
        audio_event - sound class (e.g. ``"Speech"`` / ``"BGM"``), or None
    """
    tags = _SENSEVOICE_TAG_RE.findall(raw_text)

    language: str | None = None
    emotion: str | None = None
    audio_event: str | None = None

    for tag in tags:
        if tag in _EMOTION_MAP:
            emotion = _EMOTION_MAP[tag]
        elif tag in _AUDIO_EVENTS:
            audio_event = tag
        elif tag in ("withitn", "woitn"):
            pass  # ITN flag — informational only
        else:
            # Treat as a language code (e.g. "zh", "en", "yue") — only accept
            # values that normalize successfully; unknown tags are silently ignored.
            normalized = _normalize_language(tag)
            if normalized is not None:
                language = normalized

    # Strip all tags to get clean transcript
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        clean_text = rich_transcription_postprocess(raw_text)
    except Exception:
        clean_text = _SENSEVOICE_TAG_RE.sub("", raw_text)

    return clean_text.strip(), language, emotion, audio_event


def _strip_sensevoice_tags(text: str) -> str:
    """Remove SenseVoice tags from output text (public helper for tests/external use)."""
    _, _, _, _ = _parse_sensevoice_output(text)  # parse and discard metadata
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        return str(rich_transcription_postprocess(text))
    except Exception:
        return _SENSEVOICE_TAG_RE.sub("", text)


class SenseVoiceAsrConfig(OperatorConfig):
    model: str = "iic/SenseVoiceSmall"
    language: str = "auto"  # "zh", "en", "ja", "ko", "yue", or "auto"


@register_operator
class SenseVoiceAsrOperator(Operator):
    """Transcribe audio using SenseVoice (FunASR).

    SenseVoice supports Chinese, English, Japanese, Korean, and Cantonese.
    The ``SenseVoiceSmall`` model is fast and accurate for these languages.

    In addition to the transcript, each Supervision carries:
    - ``supervision.language`` — model-detected language code
    - ``supervision.custom["emotion"]`` — per-utterance emotion label
    - ``supervision.custom["audio_event"]`` — ``"Speech"`` / ``"BGM"`` / ``"noise"``
    """

    name = "sensevoice_asr"
    config_cls = SenseVoiceAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["funasr"]

    def setup(self) -> None:
        import torch
        from funasr import AutoModel

        assert isinstance(self.config, SenseVoiceAsrConfig)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel(
            model=self.config.model,
            device=device,
            disable_update=True,
        )

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SenseVoiceAsrConfig)
        out = []
        for cut in cuts:
            audio, _sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            result = self._model.generate(
                input=audio,
                input_len=len(audio),
                language=self.config.language,
                use_itn=True,
                batch_size_s=0,
            )

            new_sups: list[Supervision] = []
            if result and len(result) > 0:
                for item in result:
                    raw = item.get("text", "") if isinstance(item, dict) else str(item)
                    clean_text, detected_lang, emotion, audio_event = _parse_sensevoice_output(raw)
                    if not clean_text:
                        continue

                    sup_custom: dict[str, str] = {}
                    if emotion is not None:
                        sup_custom["emotion"] = emotion
                    if audio_event is not None:
                        sup_custom["audio_event"] = audio_event

                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__{self.ctx.stage_name}_{len(new_sups)}",
                            recording_id=cut.recording_id,
                            start=cut.start,
                            duration=cut.duration,
                            text=clean_text,
                            language=detected_lang
                            or (
                                _normalize_language(self.config.language)
                                if self.config.language != "auto"
                                else None
                            ),
                            custom=sup_custom,
                        )
                    )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
