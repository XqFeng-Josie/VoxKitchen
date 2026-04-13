"""Standalone tool functions for common speech processing tasks.

These are thin wrappers around VoxKitchen operators that hide the
CutSet/RunContext plumbing. Use them when you just want to process
one file without building a full pipeline.

Example::

    from voxkitchen.tools import transcribe, detect_speech, estimate_snr

    text = transcribe("audio.wav", model="tiny")
    segments = detect_speech("audio.wav", method="silero")
    snr = estimate_snr("audio.wav")
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file

if TYPE_CHECKING:
    from voxkitchen.pipeline.context import RunContext


@dataclass
class SpeechSegment:
    """A detected speech segment with start/end times and optional text."""

    start: float
    end: float
    text: str = ""
    speaker: str = ""
    language: str = ""


def _make_cut(audio_path: Path) -> Cut:
    """Create a minimal Cut from an audio file path."""
    rec = recording_from_file(audio_path)
    return Cut(
        id=rec.id,
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
        recording=rec,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="tools",
            stage_name="standalone",
            created_at=datetime.now(tz=timezone.utc),
            pipeline_run_id="standalone",
        ),
    )


def _make_ctx() -> RunContext:
    """Create a minimal RunContext for standalone use."""
    from voxkitchen.pipeline.context import RunContext

    return RunContext(
        work_dir=Path(tempfile.mkdtemp(prefix="vkit-tools-")),
        pipeline_run_id="standalone",
        stage_index=0,
        stage_name="tools",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="keep",
        device="cpu",
    )


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    *,
    model: str = "tiny",
    language: str | None = None,
    beam_size: int = 5,
    compute_type: str = "int8",
) -> list[SpeechSegment]:
    """Transcribe an audio file using faster-whisper.

    Returns a list of SpeechSegments with text, start, end, and language.

    Example::

        segments = transcribe("interview.wav", model="base", language="en")
        for seg in segments:
            print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text}")
    """
    from voxkitchen.operators.annotate.faster_whisper_asr import (
        FasterWhisperAsrConfig,
        FasterWhisperAsrOperator,
    )

    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()
    config = FasterWhisperAsrConfig(
        model=model,
        language=language,
        beam_size=beam_size,
        compute_type=compute_type,
    )
    op = FasterWhisperAsrOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    return [
        SpeechSegment(
            start=s.start,
            end=s.start + s.duration,
            text=s.text or "",
            language=s.language or "",
        )
        for s in out_cut.supervisions
    ]


# ---------------------------------------------------------------------------
# VAD / Speech Detection
# ---------------------------------------------------------------------------


def detect_speech(
    audio_path: str | Path,
    *,
    method: str = "silero",
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
) -> list[SpeechSegment]:
    """Detect speech segments in an audio file.

    Args:
        method: "silero" (GPU-capable) or "webrtc" (CPU-only, lightweight).
        threshold: Detection threshold (silero only).

    Returns a list of SpeechSegments with start/end times (no text).

    Example::

        segments = detect_speech("recording.wav", method="silero")
        print(f"Found {len(segments)} speech segments")
    """
    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()

    if method == "silero":
        from voxkitchen.operators.segment.silero_vad import (
            SileroVadConfig,
            SileroVadOperator,
        )

        config = SileroVadConfig(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
        )
        op = SileroVadOperator(config, ctx)
    elif method == "webrtc":
        from voxkitchen.operators.segment.webrtc_vad import (
            WebrtcVadConfig,
            WebrtcVadOperator,
        )

        config = WebrtcVadConfig(min_speech_duration_ms=min_speech_duration_ms)  # type: ignore[assignment]
        op = WebrtcVadOperator(config, ctx)  # type: ignore[assignment]
    else:
        raise ValueError(f"unknown VAD method: {method!r}, use 'silero' or 'webrtc'")

    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    return [SpeechSegment(start=c.start, end=c.start + c.duration) for c in result]


# ---------------------------------------------------------------------------
# Quality Estimation
# ---------------------------------------------------------------------------


def estimate_snr(audio_path: str | Path) -> float:
    """Estimate the signal-to-noise ratio of an audio file in dB.

    Example::

        snr = estimate_snr("noisy_recording.wav")
        print(f"SNR: {snr:.1f} dB")
    """
    from voxkitchen.operators.quality.snr_estimate import (
        SnrEstimateConfig,
        SnrEstimateOperator,
    )

    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()
    op = SnrEstimateOperator(SnrEstimateConfig(), ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    return out_cut.metrics.get("snr", 0.0)


# ---------------------------------------------------------------------------
# Audio Processing
# ---------------------------------------------------------------------------


def resample_audio(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    target_sr: int = 16000,
    target_channels: int | None = 1,
) -> Path:
    """Resample an audio file and save to output_path.

    Example::

        resample_audio("input.wav", "output_16k.wav", target_sr=16000)
    """
    from voxkitchen.operators.basic.resample import ResampleConfig, ResampleOperator

    in_path = Path(audio_path)
    out_path = Path(output_path)
    cut = _make_cut(in_path)
    ctx = _make_ctx()
    config = ResampleConfig(target_sr=target_sr, target_channels=target_channels)
    op = ResampleOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    # Copy the derived file to the requested output path
    out_cut = next(iter(result))
    if out_cut.recording:
        derived = Path(out_cut.recording.sources[0].source)
        import shutil

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived, out_path)
    return out_path


def normalize_loudness(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    target_lufs: float = -23.0,
) -> Path:
    """Normalize the loudness of an audio file to a target LUFS.

    Example::

        normalize_loudness("loud.wav", "normalized.wav", target_lufs=-23.0)
    """
    from voxkitchen.operators.basic.loudness_normalize import (
        LoudnessNormalizeConfig,
        LoudnessNormalizeOperator,
    )

    in_path = Path(audio_path)
    out_path = Path(output_path)
    cut = _make_cut(in_path)
    ctx = _make_ctx()
    config = LoudnessNormalizeConfig(target_lufs=target_lufs)
    op = LoudnessNormalizeOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    if out_cut.recording:
        derived = Path(out_cut.recording.sources[0].source)
        import shutil

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(derived, out_path)
    return out_path
