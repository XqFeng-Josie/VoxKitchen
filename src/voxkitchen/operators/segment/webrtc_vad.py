"""WebrtcVad operator: speech-activity detection via webrtcvad."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut
from voxkitchen.utils.time import now_utc

_WEBRTCVAD_RATES = frozenset({8000, 16000, 32000, 48000})
_TARGET_SR = 16000


class WebrtcVadConfig(OperatorConfig):
    aggressiveness: int = 2  # 0-3
    frame_duration_ms: int = 30  # 10, 20, or 30
    min_speech_duration_ms: int = 250
    padding_ms: int = 30


@register_operator
class WebrtcVadOperator(Operator):
    """Detect speech regions using webrtcvad and emit one child Cut per region.

    Reads audio bytes from the parent Cut, runs frame-by-frame VAD, merges
    consecutive speech frames, applies minimum-duration and padding, then
    creates child Cuts for each speech region.  No new audio is written.
    """

    name = "webrtc_vad"
    config_cls = WebrtcVadConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = True
    required_extras: ClassVar[list[str]] = ["segment"]

    def setup(self) -> None:
        import webrtcvad

        self._webrtcvad = webrtcvad

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, WebrtcVadConfig)
        out: list[Cut] = []
        for cut in cuts:
            out.extend(self._segment_cut(cut))
        return CutSet(out)

    def _segment_cut(self, cut: Cut) -> list[Cut]:
        assert isinstance(self.config, WebrtcVadConfig)
        audio, sr = load_audio_for_cut(cut)

        # Ensure mono
        if audio.ndim == 2:
            audio = audio[:, 0]

        # Resample to a webrtcvad-supported rate if needed
        if sr not in _WEBRTCVAD_RATES:
            try:
                import torch
                import torchaudio

                t = torchaudio.functional.resample(
                    torch.from_numpy(audio).unsqueeze(0), sr, _TARGET_SR
                ).squeeze(0)
                audio = t.numpy()
            except ImportError:
                from scipy.signal import resample as scipy_resample

                new_len = int(len(audio) * _TARGET_SR / sr)
                audio = scipy_resample(audio, new_len).astype(np.float32)
            sr = _TARGET_SR

        # Convert float32 → int16 PCM
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        pcm_bytes = pcm.tobytes()

        frame_ms = self.config.frame_duration_ms
        frame_samples = sr * frame_ms // 1000
        frame_bytes = frame_samples * 2  # 2 bytes per int16 sample

        vad = self._webrtcvad.Vad(self.config.aggressiveness)

        # Collect per-frame speech labels
        speech_flags: list[bool] = []
        offset = 0
        while offset + frame_bytes <= len(pcm_bytes):
            frame = pcm_bytes[offset : offset + frame_bytes]
            is_speech = vad.is_speech(frame, sr)
            speech_flags.append(is_speech)
            offset += frame_bytes

        if not speech_flags:
            return []

        # Merge consecutive speech frames into [start_ms, end_ms] regions
        regions: list[tuple[float, float]] = []
        in_speech = False
        region_start = 0
        for i, flag in enumerate(speech_flags):
            if flag and not in_speech:
                region_start = i * frame_ms
                in_speech = True
            elif not flag and in_speech:
                regions.append((region_start, i * frame_ms))
                in_speech = False
        if in_speech:
            regions.append((region_start, len(speech_flags) * frame_ms))

        # Filter by min_speech_duration and expand by padding_ms
        min_ms = self.config.min_speech_duration_ms
        pad_ms = self.config.padding_ms
        total_ms = len(audio) / sr * 1000.0

        filtered: list[tuple[float, float]] = []
        for start_ms, end_ms in regions:
            if (end_ms - start_ms) < min_ms:
                continue
            padded_start = max(0.0, start_ms - pad_ms)
            padded_end = min(total_ms, end_ms + pad_ms)
            filtered.append((padded_start, padded_end))

        # Build child Cuts
        generated_by = f"webrtc_vad@agg{self.config.aggressiveness}"
        stage_name = getattr(getattr(self, "ctx", None), "stage_name", "unknown")
        pipeline_run_id = getattr(getattr(self, "ctx", None), "pipeline_run_id", "unknown")

        children: list[Cut] = []
        for idx, (start_ms, end_ms) in enumerate(filtered):
            start_sec = cut.start + start_ms / 1000.0
            duration_sec = (end_ms - start_ms) / 1000.0
            children.append(
                Cut(
                    id=f"{cut.id}__vad{idx}",
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
