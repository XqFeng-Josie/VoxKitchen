"""ChatTTS operator: conversational-style text-to-speech synthesis.

ChatTTS produces natural-sounding conversational speech. Supports
speaker sampling via seed for reproducibility. No voice cloning.
Output sample rate is 24kHz. Requires GPU (4GB+ VRAM).

Prosody control tokens: ``[laugh]``, ``[uv_break]``, ``[lbreak]``
can be embedded directly in the input text.
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

CHATTTS_SR = 24000


class TtsChatTTSConfig(OperatorConfig):
    seed: int | None = None  # fix speaker timbre; None = random
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20


@register_operator
class TtsChatTTSOperator(Operator):
    """Synthesize conversational speech using ChatTTS."""

    name = "tts_chattts"
    config_cls = TtsChatTTSConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-chattts"]

    _chat: Any
    _spk: Any

    def setup(self) -> None:
        import ChatTTS
        import torch

        assert isinstance(self.config, TtsChatTTSConfig)
        self._chat = ChatTTS.Chat()
        self._chat.load(compile=False)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            self._spk = self._chat.sample_random_speaker()
        else:
            self._spk = None

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, TtsChatTTSConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            params_infer = self._chat.InferCodeParams(
                spk_emb=self._spk,
                temperature=self.config.temperature,
                top_P=self.config.top_p,
                top_K=self.config.top_k,
            )

            wavs = self._chat.infer(
                [text],
                params_infer_code=params_infer,
            )

            if wavs is None or len(wavs) == 0:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.asarray(wavs[0], dtype=np.float32).flatten()
            audio = np.clip(audio, -1.0, 1.0)

            out_path = derived_dir / f"{cut.id}__chattts.wav"
            save_audio(out_path, audio, CHATTTS_SR)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_chattts")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__chattts",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_chattts",
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
        self._chat = None
        self._spk = None
