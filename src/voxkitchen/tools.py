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
# Audio Info
# ---------------------------------------------------------------------------


@dataclass
class AudioInfo:
    """Audio file properties."""

    path: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration: float
    format: str
    real_sample_rate: float | None = None  # estimated from spectral content


def audio_info(
    audio_path: str | Path,
    *,
    estimate_real_sr: bool = False,
) -> AudioInfo:
    """Get audio file properties: sample rate, channels, duration, format.

    Args:
        estimate_real_sr: If True, estimate the real (effective) sample rate
            by analyzing the spectral content. This detects files that were
            upsampled from a lower rate (e.g., an 8kHz recording saved as
            16kHz WAV). Requires torch.

    Example::

        info = audio_info("recording.wav")
        print(f"{info.sample_rate} Hz, {info.num_channels} ch, {info.duration:.1f}s")

        # Detect upsampled audio
        info = audio_info("suspicious.wav", estimate_real_sr=True)
        print(f"Header says {info.sample_rate} Hz, real content is {info.real_sample_rate} Hz")
    """
    import soundfile as sf

    path = Path(audio_path)
    si = sf.info(str(path))
    real_sr: float | None = None

    if estimate_real_sr:
        real_sr = estimate_bandwidth(path)

    return AudioInfo(
        path=str(path),
        sample_rate=si.samplerate,
        num_channels=si.channels,
        num_samples=si.frames,
        duration=si.duration,
        format=si.format,
        real_sample_rate=real_sr,
    )


def estimate_bandwidth(audio_path: str | Path) -> float:
    """Estimate the real (effective) sample rate of an audio file.

    Detects files that were upsampled from a lower rate — e.g., an 8 kHz
    telephone recording saved as 16 kHz WAV will return ≈ 8000.

    This is a convenience wrapper around the ``bandwidth_estimate`` operator.

    Returns:
        Estimated real sample rate in Hz (= effective bandwidth x 2).

    Example::

        real_sr = estimate_bandwidth("maybe_upsampled.wav")
        print(f"Real sample rate: {real_sr:.0f} Hz")
    """
    from voxkitchen.operators.quality.bandwidth_estimate import (
        BandwidthEstimateConfig,
        BandwidthEstimateOperator,
    )

    cut = _make_cut(Path(audio_path))
    ctx = _make_ctx()
    op = BandwidthEstimateOperator(BandwidthEstimateConfig(), ctx)
    op.setup()
    result = next(iter(op.process(CutSet([cut]))))
    return result.metrics.get("bandwidth_khz", 0.0) * 1000  # kHz → Hz


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str | Path,
    *,
    engine: str = "faster-whisper",
    model: str = "tiny",
    language: str | None = None,
    beam_size: int = 5,
    compute_type: str = "int8",
) -> list[SpeechSegment]:
    """Transcribe an audio file.

    Args:
        engine: ASR engine to use. Options:

            - ``"faster-whisper"`` — OpenAI Whisper via CTranslate2 (default)
            - ``"sensevoice"`` — Alibaba SenseVoice (multi-language, via FunASR)
            - ``"paraformer"`` — Alibaba Paraformer (fast, Chinese-optimized, via FunASR)
            - ``"wenet"`` — WeNet (production-grade Chinese/English)

        model: Model name or path. Defaults depend on engine:

            - faster-whisper: "tiny", "base", "small", "medium", "large-v3"
            - sensevoice: "iic/SenseVoiceSmall"
            - paraformer: "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
            - wenet: "chinese", "english"

        language: Language hint (None = auto-detect where supported).
        beam_size: Beam size (faster-whisper only).
        compute_type: "int8" (CPU) or "float16" (GPU). faster-whisper only.

    Returns:
        List of SpeechSegments with text, start, end, and language.

    Example::

        # Whisper (English/multilingual)
        segments = transcribe("interview.wav", model="base", language="en")

        # SenseVoice (Chinese/multilingual, fast)
        segments = transcribe("meeting.wav", engine="sensevoice")

        # Paraformer (Chinese, very fast, with punctuation)
        segments = transcribe("lecture.wav", engine="paraformer")

        # WeNet (Chinese, production-grade)
        segments = transcribe("call.wav", engine="wenet", model="chinese")
    """
    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()

    if engine == "faster-whisper":
        from voxkitchen.operators.annotate.faster_whisper_asr import (
            FasterWhisperAsrConfig,
            FasterWhisperAsrOperator,
        )

        config = FasterWhisperAsrConfig(
            model=model,
            language=language,
            beam_size=beam_size,
            compute_type=compute_type,
        )
        op = FasterWhisperAsrOperator(config, ctx)

    elif engine == "sensevoice":
        from voxkitchen.operators.annotate.sensevoice_asr import (
            SenseVoiceAsrConfig,
            SenseVoiceAsrOperator,
        )

        sv_model = model if model != "tiny" else "iic/SenseVoiceSmall"
        config = SenseVoiceAsrConfig(model=sv_model, language=language or "auto")  # type: ignore[assignment]
        op = SenseVoiceAsrOperator(config, ctx)  # type: ignore[assignment]

    elif engine == "paraformer":
        from voxkitchen.operators.annotate.paraformer_asr import (
            ParaformerAsrConfig,
            ParaformerAsrOperator,
        )

        pf_model = (
            model
            if model != "tiny"
            else "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
        config = ParaformerAsrConfig(model=pf_model, language=language or "zh")  # type: ignore[assignment]
        op = ParaformerAsrOperator(config, ctx)  # type: ignore[assignment]

    elif engine == "wenet":
        from voxkitchen.operators.annotate.wenet_asr import (
            WenetAsrConfig,
            WenetAsrOperator,
        )

        wn_model = model if model != "tiny" else "chinese"
        config = WenetAsrConfig(model=wn_model, language=language or "zh")  # type: ignore[assignment]
        op = WenetAsrOperator(config, ctx)  # type: ignore[assignment]

    else:
        raise ValueError(
            f"unknown ASR engine: {engine!r}. "
            f"Options: 'faster-whisper', 'sensevoice', 'paraformer', 'wenet'"
        )

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
# Gender Classification
# ---------------------------------------------------------------------------


def classify_gender(
    audio_path: str | Path,
    *,
    method: str = "f0",
    f0_threshold: float = 165.0,
) -> dict[str, str | float | None]:
    """Classify the speaker's gender from an audio file.

    Args:
        method: Detection method. Options:

            - ``"f0"`` — Pitch-based (librosa pyin). Fastest, no model, ~80-85%.
            - ``"speechbrain"`` — SpeechBrain classifier. Needs model download, ~95%.
            - ``"inaspeechsegmenter"`` — INA's segmenter. Needs tensorflow, ~90-95%.

        f0_threshold: Hz boundary for the F0 method (default 165).

    Returns:
        Dict with ``"gender"`` ("m"/"f"/"o"), ``"method"``, and method-specific details.

    Example::

        result = classify_gender("speaker.wav")
        print(result["gender"])  # "m" or "f"
        print(result["median_f0"])  # 120.5 (if method="f0")

        result = classify_gender("speaker.wav", method="inaspeechsegmenter")
    """
    from voxkitchen.operators.annotate.gender_classify import (
        GenderClassifyConfig,
        GenderClassifyOperator,
    )

    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()
    config = GenderClassifyConfig(method=method, f0_threshold=f0_threshold)
    op = GenderClassifyOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    # Find the gender supervision
    for s in out_cut.supervisions:
        if s.gender:
            return {"gender": s.gender, **s.custom}
    return {"gender": "o", "method": method}


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


# ---------------------------------------------------------------------------
# Speaker Embedding
# ---------------------------------------------------------------------------


def extract_speaker_embedding(
    audio_path: str | Path,
    *,
    method: str = "wespeaker",
    model: str = "english",
) -> list[float]:
    """Extract a speaker embedding vector from an audio file.

    Args:
        method: "wespeaker" (default) or "speechbrain".
        model: Model name. Defaults depend on method:

            - wespeaker: "english", "chinese", etc.
            - speechbrain: "speechbrain/spkrec-ecapa-voxceleb"

    Returns:
        Speaker embedding as a list of floats (e.g. 512-d for WeSpeaker).

    Example::

        emb = extract_speaker_embedding("speaker.wav")
        print(f"Embedding dim: {len(emb)}")  # 512

        # SpeechBrain backend
        emb = extract_speaker_embedding("speaker.wav", method="speechbrain",
                                         model="speechbrain/spkrec-ecapa-voxceleb")
    """
    from voxkitchen.operators.annotate.speaker_embed import (
        SpeakerEmbedConfig,
        SpeakerEmbedOperator,
    )

    path = Path(audio_path)
    cut = _make_cut(path)
    ctx = _make_ctx()
    config = SpeakerEmbedConfig(
        method=method,
        wespeaker_model=model if method == "wespeaker" else "english",
        speechbrain_model=model
        if method == "speechbrain"
        else "speechbrain/spkrec-ecapa-voxceleb",
    )
    op = SpeakerEmbedOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    return out_cut.custom.get("speaker_embedding", [])


# ---------------------------------------------------------------------------
# Speech Enhancement
# ---------------------------------------------------------------------------


def enhance_speech(
    audio_path: str | Path,
    output_path: str | Path,
    *,
    aggressiveness: float = 0.5,
) -> Path:
    """Remove background noise from an audio file using DeepFilterNet.

    Args:
        aggressiveness: 0.0 (light) to 1.0 (aggressive denoising).

    Example::

        enhance_speech("noisy.wav", "clean.wav", aggressiveness=0.5)
    """
    from voxkitchen.operators.annotate.speech_enhance import (
        SpeechEnhanceConfig,
        SpeechEnhanceOperator,
    )

    in_path = Path(audio_path)
    out_path = Path(output_path)
    cut = _make_cut(in_path)
    ctx = _make_ctx()
    config = SpeechEnhanceConfig(aggressiveness=aggressiveness)
    op = SpeechEnhanceOperator(config, ctx)
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


# ---------------------------------------------------------------------------
# Forced Alignment
# ---------------------------------------------------------------------------


def align_words(
    audio_path: str | Path,
    text: str,
    *,
    language: str = "eng",
) -> list[dict[str, object]]:
    """Align text to audio at word level using CTC forced alignment.

    Args:
        text: The text to align (e.g. from a prior ASR transcription).
        language: ISO 639-3 language code (default "eng" for English).

    Returns:
        List of word alignments, each a dict with "text", "start", "end".

    Example::

        words = align_words("speech.wav", "hello world")
        for w in words:
            print(f"{w['text']}: {w['start']:.2f}s - {w['end']:.2f}s")
    """
    from voxkitchen.operators.annotate.forced_align import (
        ForcedAlignConfig,
        ForcedAlignOperator,
    )

    path = Path(audio_path)
    cut = _make_cut(path)
    # Add text as a supervision so the operator can find it
    from voxkitchen.schema.supervision import Supervision

    cut = cut.model_copy(
        update={
            "supervisions": [
                Supervision(
                    id=f"sup-{cut.id}",
                    recording_id=cut.recording_id,
                    start=0.0,
                    duration=cut.duration,
                    text=text,
                )
            ]
        }
    )
    ctx = _make_ctx()
    config = ForcedAlignConfig(language=language)
    op = ForcedAlignOperator(config, ctx)
    op.setup()
    try:
        result = op.process(CutSet([cut]))
    finally:
        op.teardown()

    out_cut = next(iter(result))
    return out_cut.custom.get("word_alignments", [])
