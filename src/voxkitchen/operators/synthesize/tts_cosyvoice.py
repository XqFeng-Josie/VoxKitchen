"""CosyVoice2 TTS operator: high-quality text-to-speech with voice cloning.

Supports three modes:
- ``sft``: built-in speaker voices (fastest, no reference audio needed)
- ``zero_shot``: clone any voice from a reference audio + transcript
- ``cross_lingual``: clone voice across languages (reference audio only)

Model is auto-downloaded from ModelScope on first use.
Output sample rate is 24kHz. Requires GPU.
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

COSYVOICE_SR = 24000


class TtsCosyVoiceConfig(OperatorConfig):
    model_id: str = "FunAudioLLM/CosyVoice2-0.5B"
    mode: str = "sft"  # sft / zero_shot / cross_lingual
    spk_id: str = "default"  # speaker ID for sft mode
    reference_audio: str | None = None  # path for zero_shot / cross_lingual
    reference_text: str | None = None  # transcript of reference audio (zero_shot)


@register_operator
class TtsCosyVoiceOperator(Operator):
    """Synthesize speech using CosyVoice2 with optional voice cloning."""

    name = "tts_cosyvoice"
    config_cls = TtsCosyVoiceConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-cosyvoice"]

    _model: Any
    _sample_rate: int

    def setup(self) -> None:
        from modelscope import snapshot_download

        assert isinstance(self.config, TtsCosyVoiceConfig)
        model_dir = snapshot_download(self.config.model_id)

        from cosyvoice.cli.cosyvoice import AutoModel

        self._model = AutoModel(model_dir=model_dir)
        self._sample_rate = getattr(self._model, "sample_rate", COSYVOICE_SR)

    def process(self, cuts: CutSet) -> CutSet:
        import torch

        assert isinstance(self.config, TtsCosyVoiceConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            text = self._extract_text(cut)
            if text is None:
                logger.warning("cut %s has no text, skipping", cut.id)
                continue

            chunks: list[np.ndarray] = []
            for chunk in self._infer(text):
                speech = chunk["tts_speech"]
                if isinstance(speech, torch.Tensor):
                    speech = speech.cpu().numpy()
                chunks.append(np.asarray(speech, dtype=np.float32).flatten())

            if not chunks:
                logger.warning("cut %s produced no audio, skipping", cut.id)
                continue

            audio = np.concatenate(chunks)
            audio = np.clip(audio, -1.0, 1.0)

            out_path = derived_dir / f"{cut.id}__cosyvoice.wav"
            save_audio(out_path, audio, self._sample_rate)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.id}_cosyvoice")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__cosyvoice",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    custom=dict(cut.custom) if cut.custom else {},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by="tts_cosyvoice",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                )
            )
        return CutSet(out_cuts)

    def _infer(self, text: str) -> Any:
        assert isinstance(self.config, TtsCosyVoiceConfig)
        mode = self.config.mode

        if mode == "sft":
            return self._model.inference_sft(text, spk_id=self.config.spk_id, stream=False)

        if mode == "zero_shot":
            if not self.config.reference_audio or not self.config.reference_text:
                raise ValueError("zero_shot mode requires both reference_audio and reference_text")
            return self._model.inference_zero_shot(
                text,
                self.config.reference_text,
                self.config.reference_audio,
                stream=False,
            )

        if mode == "cross_lingual":
            if not self.config.reference_audio:
                raise ValueError("cross_lingual mode requires reference_audio")
            return self._model.inference_cross_lingual(
                text,
                self.config.reference_audio,
                stream=False,
            )

        raise ValueError(f"unknown mode: {mode!r}, use 'sft', 'zero_shot', or 'cross_lingual'")

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        self._model = None
