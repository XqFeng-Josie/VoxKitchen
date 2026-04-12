"""Unit tests for silero_vad operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


@pytest.fixture
def speech_like_wav_16k(tmp_path: Path) -> Path:
    """Generate a 1-second voiced signal (harmonic at 150 Hz) that Silero detects as speech.

    A pure sine wave is correctly rejected by Silero as non-speech.  This
    fixture uses a harmonic series (multiple overtones above a 150 Hz
    fundamental) which resembles a voiced vowel and reliably triggers the VAD.
    """
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    f0 = 150
    audio = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
    audio = (audio / np.max(np.abs(audio)) * 0.5).astype(np.float32)
    path = tmp_path / "speech_like_16k.wav"
    sf.write(path, audio, sr)
    return path


def _make_cut(path: Path) -> Cut:
    """Build a Cut from an audio file path."""
    info = sf.info(str(path))
    rec = Recording(
        id=path.stem,
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=info.samplerate,
        num_samples=info.frames,
        duration=info.duration,
        num_channels=info.channels,
    )
    return Cut(
        id=path.stem,
        recording_id=rec.id,
        start=0.0,
        duration=info.duration,
        recording=rec,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


def test_silero_vad_is_registered() -> None:
    from voxkitchen.operators.segment.silero_vad import SileroVadOperator

    assert get_operator("silero_vad") is SileroVadOperator


def test_silero_vad_class_attrs() -> None:
    from voxkitchen.operators.segment.silero_vad import SileroVadOperator

    assert SileroVadOperator.device == "gpu"
    assert SileroVadOperator.produces_audio is False


@pytest.mark.slow
def test_silero_vad_detects_speech(speech_like_wav_16k: Path) -> None:
    """Real model on CPU: voiced harmonic signal should produce >= 1 segment."""
    from voxkitchen.operators.segment.silero_vad import SileroVadConfig, SileroVadOperator

    cut = _make_cut(speech_like_wav_16k)
    config = SileroVadConfig(
        threshold=0.3,
        min_speech_duration_ms=50,
        min_silence_duration_ms=50,
        speech_pad_ms=30,
    )
    op = SileroVadOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))

    assert len(result) >= 1


@pytest.mark.slow
def test_silero_vad_child_cuts_have_provenance(speech_like_wav_16k: Path) -> None:
    """Child cuts should have source_cut_id pointing to parent."""
    from voxkitchen.operators.segment.silero_vad import SileroVadConfig, SileroVadOperator

    cut = _make_cut(speech_like_wav_16k)
    config = SileroVadConfig(
        threshold=0.3,
        min_speech_duration_ms=50,
        min_silence_duration_ms=50,
        speech_pad_ms=30,
    )
    op = SileroVadOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))

    assert len(result) >= 1
    for child in result:
        assert child.provenance.source_cut_id == cut.id
        assert child.recording_id == cut.recording_id
        assert child.recording == cut.recording
        assert child.supervisions == []
        assert child.start >= 0.0
        assert child.duration > 0.0
