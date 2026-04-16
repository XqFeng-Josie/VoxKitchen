"""WeNet ASR operator: production-grade ASR via WeNet.

WeNet (https://github.com/wenet-e2e/wenet) is a production-first end-to-end
speech recognition toolkit. It provides pretrained models for Chinese and
English with excellent accuracy and real-time performance.

Requires: ``pip install wenet``
"""

from __future__ import annotations

from typing import ClassVar

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.supervision import Supervision
from voxkitchen.utils.audio import load_audio_for_cut


class WenetAsrConfig(OperatorConfig):
    model: str = "chinese"  # "chinese", "english", or a model dir path
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

            # WeNet expects the audio file path or wav bytes.
            # Use the file path from the recording for best compatibility.
            if cut.recording and cut.recording.sources:
                audio_path = cut.recording.sources[0].source
                result = self._decoder.decode(audio_path)
            else:
                # Fallback: save to temp file
                import tempfile

                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    sf.write(f.name, audio, sr)
                    result = self._decoder.decode(f.name)

            new_sups: list[Supervision] = []
            if result:
                # WeNet returns a string or dict depending on version
                text = result if isinstance(result, str) else result.get("text", str(result))
                text = text.strip()
                if text:
                    new_sups.append(
                        Supervision(
                            id=f"{cut.id}__wenet_0",
                            recording_id=cut.recording_id,
                            start=cut.start,
                            duration=cut.duration,
                            text=text,
                            language=self.config.language,
                        )
                    )

            updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
            out.append(updated)
        return CutSet(out)
