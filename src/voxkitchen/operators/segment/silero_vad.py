"""SileroVad operator: GPU-capable speech-activity detection via Silero VAD."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.time import now_utc

_SILERO_SR = 16000


class SileroVadConfig(OperatorConfig):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30


@register_operator
class SileroVadOperator(Operator):
    """Detect speech regions using Silero VAD and emit one child Cut per region.

    Loads the Silero VAD model via torch.hub (cached after first download).
    Works on both GPU and CPU. Requires network on first run to download
    the model (~2 MB). Use ``webrtc_vad`` or ``silence_split`` if torch
    is not available.
    """

    name = "silero_vad"
    config_cls = SileroVadConfig
    device = "gpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["segment"]

    def setup(self) -> None:
        import sys

        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load silero-vad: prefer local cache (avoids network hangs),
        # inject the repo into sys.path and import directly.
        local_repo = self._find_cached_repo("snakers4_silero-vad_master")
        if local_repo:
            repo_src = local_repo / "src"
            for p in [str(local_repo), str(repo_src)]:
                if p not in sys.path:
                    sys.path.insert(0, p)
            from hubconf import silero_vad  # type: ignore[import-not-found]

            self._model, self._utils = silero_vad(onnx=False)
        else:
            self._model, self._utils = torch.hub.load(  # type: ignore[no-untyped-call]
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
        self._model.to(self._device)

    @staticmethod
    def _find_cached_repo(repo_dir: str) -> Path | None:
        """Search common torch hub cache locations for a cached repo."""
        import os

        import torch

        hub_dir = Path(torch.hub.get_dir())
        candidates = [
            hub_dir / repo_dir,
            hub_dir.parent / repo_dir,
            Path.home() / ".cache" / "torch" / "hub" / repo_dir,
        ]
        torch_home = os.environ.get("TORCH_HOME")
        if torch_home:
            candidates.append(Path(torch_home) / "hub" / repo_dir)

        for p in candidates:
            if p.exists() and (p / "hubconf.py").exists():
                return p
        return None

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, SileroVadConfig)
        out: list[Cut] = []
        for cut in cuts:
            out.extend(self._segment_cut(cut))
        return CutSet(out)

    def _segment_cut(self, cut: Cut) -> list[Cut]:
        import torch

        assert isinstance(self.config, SileroVadConfig)
        audio, sr = load_audio_for_cut(cut)

        # Ensure mono (take channel 0 if stereo)
        if audio.ndim == 2:
            audio = audio[:, 0]

        # Resample to 16kHz if needed (Silero requires 16kHz)
        audio_tensor = torch.from_numpy(audio)
        if sr != _SILERO_SR:
            try:
                import torchaudio

                audio_tensor = torchaudio.functional.resample(
                    audio_tensor.unsqueeze(0), sr, _SILERO_SR
                ).squeeze(0)
            except ImportError:
                from scipy.signal import resample as scipy_resample

                new_len = int(len(audio) * _SILERO_SR / sr)
                audio_tensor = torch.from_numpy(scipy_resample(audio, new_len).astype(np.float32))

        audio_tensor = audio_tensor.to(self._device)

        # get_speech_timestamps is index 0 in the utils tuple
        get_speech_timestamps = self._utils[0]
        timestamps = get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=False,
        )

        if not timestamps:
            return []

        generated_by = f"silero_vad@thr{self.config.threshold}"
        stage_name = getattr(getattr(self, "ctx", None), "stage_name", "unknown")
        pipeline_run_id = getattr(getattr(self, "ctx", None), "pipeline_run_id", "unknown")

        children: list[Cut] = []
        for idx, ts in enumerate(timestamps):
            start_sec = cut.start + ts["start"] / _SILERO_SR
            end_sec = cut.start + ts["end"] / _SILERO_SR
            duration_sec = end_sec - start_sec
            children.append(
                Cut(
                    id=f"{cut.id}__svad{idx}",
                    recording_id=cut.recording_id,
                    start=start_sec,
                    duration=duration_sec,
                    recording=cut.recording,
                    supervisions=[],
                    metrics={},
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=generated_by,
                        stage_name=stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return children
