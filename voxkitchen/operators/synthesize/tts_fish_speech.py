"""Fish-Speech TTS operator: codec-LM text-to-speech with voice cloning.

Supports 13 languages. Output sample rate 44100 Hz. Requires GPU
(24 GB+ VRAM recommended).

.. warning::
   Targets fish-speech **1.x** API (``from fish_speech.inference import
   TTSInference``). Upstream reshuffled to ``TTSInferenceEngine`` in 2.0.
   Excluded from ``EXPECTED_OPERATORS["fish-speech"]``; a follow-up PR
   will rewrite ``_load_model`` / ``_infer`` against the 2.0 API.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file, save_audio
from voxkitchen.utils.time import now_utc

logger = logging.getLogger(__name__)

FISH_SPEECH_SR = 44100


class TtsFishSpeechConfig(OperatorConfig):
    model_id: str = "fishaudio/fish-speech-1.5"
    reference_audio: str | None = None
    reference_text: str | None = None
    max_new_tokens: int = 1024
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.2


@register_operator
class TtsFishSpeechOperator(Operator):
    """Synthesize speech using Fish-Speech codec language model."""

    name = "tts_fish_speech"
    config_cls = TtsFishSpeechConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-fish-speech"]

    _inference: Any
    _sample_rate: int

    def setup(self) -> None:
        assert isinstance(self.config, TtsFishSpeechConfig)
        self._sample_rate = FISH_SPEECH_SR
        self._load_model()

    def _load_model(self) -> None:
        """Load Fish-Speech inference pipeline.

        Fish-Speech packages its inference as a pipeline that handles
        VQGAN encoding/decoding and LLM generation. The exact import
        path may vary across versions -- adapt if needed.
        """
        assert isinstance(self.config, TtsFishSpeechConfig)
        try:
            from fish_speech.inference import TTSInference

            self._inference = TTSInference(model_id=self.config.model_id)
        except ImportError:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(self.config.model_id)
            from fish_speech.inference import TTSInference

            self._inference = TTSInference(model_dir=model_dir)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, TtsFishSpeechConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            audio = self._infer(text)
            if audio is None or len(audio) == 0:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

            out_path = derived_dir / f"{cut.id}__fish_speech.wav"
            save_audio(out_path, audio, self._sample_rate)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_fish_speech")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__fish_speech",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_fish_speech",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    def _infer(self, text: str) -> np.ndarray | None:
        """Run TTS inference for a single text string.

        Returns 1-D float32 numpy array of audio samples, or None on failure.
        """
        assert isinstance(self.config, TtsFishSpeechConfig)
        try:
            ref_audio = self.config.reference_audio
            ref_text = self.config.reference_text

            result = self._inference.synthesize(
                text=text,
                reference_audio=ref_audio,
                reference_text=ref_text,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
            )

            if isinstance(result, tuple):
                audio, sr = result
                self._sample_rate = sr
            else:
                audio = result

            return np.asarray(audio, dtype=np.float32).flatten()
        except Exception:
            logger.exception("Fish-Speech inference failed")
            return None

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._inference = None
