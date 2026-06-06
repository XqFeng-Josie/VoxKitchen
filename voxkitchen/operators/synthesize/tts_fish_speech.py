"""Fish-Speech TTS operator: codec-LM text-to-speech with voice cloning.

Uses the Fish-Speech S2 inference stack. Output sample rate is determined
by the bundled DAC codec. A GPU with large VRAM is strongly recommended.
"""

from __future__ import annotations

import logging
import queue
from pathlib import Path
from typing import Any, ClassVar

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
    model_id: str = "fishaudio/s2-pro"
    reference_audio: str | None = None
    reference_text: str | None = None
    max_new_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.8
    repetition_penalty: float = 1.1
    chunk_length: int = 200
    seed: int | None = None
    compile: bool = False
    half: bool = False


@register_operator
class TtsFishSpeechOperator(Operator):
    """Synthesize speech using Fish-Speech codec language model."""

    name = "tts_fish_speech"
    config_cls = TtsFishSpeechConfig
    device = "gpu"
    produces_audio = True
    reads_audio_bytes = False
    required_extras = ["tts-fish-speech"]
    reads: ClassVar[list[str]] = ["supervisions.text"]
    writes: ClassVar[list[str]] = ["audio"]

    _inference: Any
    _llama_queue: queue.Queue[Any] | None
    _sample_rate: int

    def setup(self) -> None:
        assert isinstance(self.config, TtsFishSpeechConfig)
        self._sample_rate = FISH_SPEECH_SR
        self._llama_queue = None
        self._load_model()

    def _load_model(self) -> None:
        """Load Fish-Speech S2 inference engine."""
        assert isinstance(self.config, TtsFishSpeechConfig)
        import torch
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.text2semantic.inference import (
            launch_thread_safe_queue,
            load_codec_model,
        )

        checkpoint_path = self._resolve_checkpoint_path(self.config.model_id)
        codec_checkpoint = checkpoint_path / "codec.pth"
        if not codec_checkpoint.is_file():
            raise FileNotFoundError(
                f"Fish-Speech checkpoint at {checkpoint_path} does not contain codec.pth. "
                "Use an S2 checkpoint such as 'fishaudio/s2-pro'."
            )

        device = self._select_device()
        precision = torch.float16 if self.config.half else torch.bfloat16

        self._llama_queue = launch_thread_safe_queue(
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            compile=self.config.compile,
        )
        decoder_model = load_codec_model(codec_checkpoint, device, precision)
        self._inference = TTSInferenceEngine(
            llama_queue=self._llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=self.config.compile,
        )

    @staticmethod
    def _resolve_checkpoint_path(model_id: str) -> Path:
        path = Path(model_id).expanduser()
        if path.exists():
            return path.resolve()

        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model_id)).resolve()

    def _select_device(self) -> str:
        import torch

        ctx_device = getattr(self.ctx, "device", "cpu")
        if str(ctx_device).startswith("cuda") and torch.cuda.is_available():
            return str(ctx_device)
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError(
            "tts_fish_speech requires a CUDA GPU. Run with Docker GPU access "
            "enabled, for example `vkit docker run --tag fish-speech ...` on "
            "a machine with NVIDIA Container Toolkit."
        )

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
                raise RuntimeError(f"tts_fish_speech produced no audio for cut {cut.id}")

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

    def _infer(self, text: str) -> np.ndarray[Any, Any] | None:
        """Run TTS inference for a single text string.

        Returns 1-D float32 numpy array of audio samples, or None if the model
        produced no output (caller must treat None as an error and raise).
        Raises on any inference failure.
        """
        assert isinstance(self.config, TtsFishSpeechConfig)
        import torch

        ref_audio = self.config.reference_audio
        ref_text = self.config.reference_text
        references = []
        if ref_audio:
            from fish_speech.utils.schema import ServeReferenceAudio

            references.append(
                ServeReferenceAudio(
                    audio=Path(ref_audio).expanduser().read_bytes(),
                    text=ref_text or "",
                )
            )

        from fish_speech.utils.schema import ServeTTSRequest

        req = ServeTTSRequest(
            text=text,
            references=references,
            seed=self.config.seed,
            max_new_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            chunk_length=self.config.chunk_length,
        )

        precision = torch.float16 if self.config.half else torch.bfloat16
        final_audio: np.ndarray[Any, Any] | None = None
        with torch.autocast(device_type="cuda", dtype=precision):
            for result in self._inference.inference(req):
                if result.code == "error":
                    raise RuntimeError(result.error or "Fish-Speech inference failed")
                if result.code == "final" and result.audio is not None:
                    sr, audio = result.audio
                    self._sample_rate = sr
                    final_audio = np.asarray(audio, dtype=np.float32).flatten()
                    break

        return final_audio

    @staticmethod
    def _extract_text(cut: Cut) -> str | None:
        for sup in cut.supervisions:
            if sup.text:
                return sup.text
        return None

    def teardown(self) -> None:
        if self._llama_queue is not None:
            self._llama_queue.put(None)
            self._llama_queue = None
        self._inference = None
