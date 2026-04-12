"""Unit tests for faster_whisper_asr operator."""

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

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mono_wav_16k(tmp_path: Path) -> Path:
    """1-second 440 Hz sine wave at 16 kHz, saved as WAV."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    path = tmp_path / "tone_16k.wav"
    sf.write(path, audio, sr)
    return path


def _make_cut(path: Path) -> Cut:
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
            created_at=datetime(2026, 4, 12, tzinfo=timezone.utc),
            pipeline_run_id="run-test",
        ),
    )


# ---------------------------------------------------------------------------
# Fast (no model download)
# ---------------------------------------------------------------------------


def test_faster_whisper_asr_is_registered() -> None:
    from voxkitchen.operators.annotate.faster_whisper_asr import FasterWhisperAsrOperator

    assert get_operator("faster_whisper_asr") is FasterWhisperAsrOperator


def test_faster_whisper_asr_class_attrs() -> None:
    from voxkitchen.operators.annotate.faster_whisper_asr import FasterWhisperAsrOperator

    assert FasterWhisperAsrOperator.device == "gpu"
    assert FasterWhisperAsrOperator.produces_audio is False
    assert FasterWhisperAsrOperator.reads_audio_bytes is True
    assert "asr" in FasterWhisperAsrOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (downloads whisper-tiny ~75 MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_faster_whisper_asr_transcribes(mono_wav_16k: Path) -> None:
    """Real tiny model on CPU: a sine wave should complete without error.

    A pure tone won't produce meaningful speech, so we only assert the
    operator runs and returns the same number of cuts — supervisions may
    be empty for non-speech audio.
    """
    from voxkitchen.operators.annotate.faster_whisper_asr import (
        FasterWhisperAsrConfig,
        FasterWhisperAsrOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = FasterWhisperAsrConfig(model="tiny", compute_type="int8")
    op = FasterWhisperAsrOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cuts = list(result)

    # One cut in → one cut out
    assert len(out_cuts) == 1
    out_cut = out_cuts[0]
    # Supervisions list exists (may be empty for a pure sine wave)
    assert isinstance(out_cut.supervisions, list)


@pytest.mark.slow
def test_faster_whisper_asr_adds_language(mono_wav_16k: Path) -> None:
    """Any supervisions emitted should carry a language field (auto-detect)."""
    from voxkitchen.operators.annotate.faster_whisper_asr import (
        FasterWhisperAsrConfig,
        FasterWhisperAsrOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = FasterWhisperAsrConfig(model="tiny", language=None, compute_type="int8")
    op = FasterWhisperAsrOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cut = next(iter(result))

    for sup in out_cut.supervisions:
        # Each supervision should have a non-empty language tag
        assert sup.language is not None
        assert len(sup.language) > 0
