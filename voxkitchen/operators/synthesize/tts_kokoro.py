"""Kokoro TTS operator: lightweight text-to-speech synthesis.

Kokoro is a compact (82M params) TTS model supporting 8 languages.
Can run on CPU. Output sample rate is 24kHz.

Requires system dependency: ``sudo apt-get install espeak-ng``
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

KOKORO_SR = 24000


class TtsKokoroConfig(OperatorConfig):
    voice: str = "af_heart"
    lang_code: str = "a"  # a=AmE, b=BrE, j=Japanese, z=Mandarin
    speed: float = 1.0


@register_operator
class TtsKokoroOperator(Operator):
    """Synthesize speech from text using Kokoro TTS."""

    name = "tts_kokoro"
    config_cls = TtsKokoroConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-kokoro"]

    _pipeline: Any

    def setup(self) -> None:
        from kokoro import KPipeline

        assert isinstance(self.config, TtsKokoroConfig)
        self._pipeline = KPipeline(lang_code=self.config.lang_code)

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, TtsKokoroConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            audio_chunks: list[np.ndarray] = []
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.config.voice, speed=self.config.speed
            ):
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

            if not audio_chunks:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio_full = np.concatenate(audio_chunks)
            out_path = derived_dir / f"{cut.id}__kokoro.wav"
            save_audio(out_path, audio_full, KOKORO_SR)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_kokoro")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__kokoro",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_kokoro",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._pipeline = None
