"""WeNet ASR operator: production-grade ASR via WeNet.

WeNet (https://github.com/wenet-e2e/wenet) is a production-first end-to-end
speech recognition toolkit. It provides pretrained models for Chinese and
English with excellent accuracy and real-time performance.

Runtime: ``vkit docker run --tag asr <yaml>``
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.language import normalize_language


class WenetAsrConfig(OperatorConfig):
    # A wenet hub model name or a local model dir. Valid hub names (per
    # wenet.cli.hub.Hub.assets) include "wenetspeech", "paraformer",
    # "sensevoice_small", "firered", and the "whisper-*" set. The old
    # "chinese"/"english" aliases were removed upstream.
    model: str = "wenetspeech"
    language: str = "zh"


@register_operator
class WenetAsrOperator(Operator):
    """Transcribe audio using WeNet.

    WeNet supports streaming and non-streaming decoding. This operator
    uses non-streaming (offline) mode for best accuracy.
    """

    name = "wenet_asr"
    config_cls = WenetAsrConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["wenet"]
    reads: ClassVar[list[str]] = ["audio"]
    writes: ClassVar[list[str]] = ["supervisions.text"]

    def setup(self) -> None:
        import wenet

        assert isinstance(self.config, WenetAsrConfig)
        self._decoder = wenet.load_model(self.config.model)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, WenetAsrConfig)
        out = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            if audio.ndim == 2:
                audio = audio[:, 0]

            # WeNet's CLI model transcribes from a wav file path. load_model()
            # returns the raw ASRModel with `transcribe(wav)` -> a result object
            # whose `.text` is the decoded string (older versions returned a
            # str/dict; handle both).
            if cut.recording and cut.recording.sources:
                audio_path = cut.recording.sources[0].source
                result = self._decoder.transcribe(audio_path)
            else:
                # Fallback: save to temp file
                import tempfile

                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, sr)
                    result = self._decoder.transcribe(f.name)

            new_sups: list[Supervision] = []
            if result is not None:
                if hasattr(result, "text"):
                    text = result.text
                elif isinstance(result, str):
                    text = result
                elif isinstance(result, dict):
                    text = result.get("text", "")
                else:
                    text = str(result)
                text = (text or "").strip()
                if text:
                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__{self.ctx.stage_name}_0",
                            recording_id=cut.recording_id,
                            start=cut.start,
                            duration=cut.duration,
                            text=text,
                            language=normalize_language(self.config.language),
                        )
                    )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
