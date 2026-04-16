"""Unit tests for whisperx_asr operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("faster_whisper")

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


def test_whisperx_asr_is_registered() -> None:
    from voxkitchen.operators.annotate.whisperx_asr import WhisperxAsrOperator

    assert get_operator("whisperx_asr") is WhisperxAsrOperator


# ---------------------------------------------------------------------------
# Slow (downloads whisper-tiny via faster-whisper fallback ~75 MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_whisperx_asr_transcribes(mono_wav_16k: Path) -> None:
    """Should work even if whisperx is not installed (falls back to faster-whisper)."""
    from voxkitchen.operators.annotate.whisperx_asr import (
        WhisperxAsrConfig,
        WhisperxAsrOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = WhisperxAsrConfig(model="tiny", compute_type="int8")
    op = WhisperxAsrOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cuts = list(result)

    assert len(out_cuts) == 1
    assert isinstance(out_cuts[0].supervisions, list)


@pytest.mark.slow
def test_whisperx_asr_output_has_supervisions(mono_wav_16k: Path) -> None:
    """Supervisions (if any) should have expected fields set."""
    from voxkitchen.operators.annotate.whisperx_asr import (
        WhisperxAsrConfig,
        WhisperxAsrOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = WhisperxAsrConfig(model="tiny", compute_type="int8")
    op = WhisperxAsrOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cut = next(iter(result))

    for sup in out_cut.supervisions:
        assert sup.recording_id == cut.recording_id
        assert sup.duration > 0
        assert sup.language is not None
