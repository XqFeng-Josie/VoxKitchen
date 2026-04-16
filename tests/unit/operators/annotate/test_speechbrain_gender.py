"""Unit tests for speechbrain_gender operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip entire module if speechbrain is not installed.
pytest.importorskip("speechbrain")

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


def test_speechbrain_gender_is_registered() -> None:
    from voxkitchen.operators.annotate.speechbrain_gender import SpeechBrainGenderOperator

    assert get_operator("speechbrain_gender") is SpeechBrainGenderOperator


def test_speechbrain_gender_class_attrs() -> None:
    from voxkitchen.operators.annotate.speechbrain_gender import SpeechBrainGenderOperator

    assert SpeechBrainGenderOperator.device == "gpu"
    assert SpeechBrainGenderOperator.produces_audio is False
    assert SpeechBrainGenderOperator.reads_audio_bytes is True
    assert "classify" in SpeechBrainGenderOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (may download model or run as no-op)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_speechbrain_gender_runs_without_crash(mono_wav_16k: Path) -> None:
    """Model may not be a true gender classifier. Just verify no crash.

    The default model (spkrec-ecapa-voxceleb) is a speaker-recognition model
    and may or may not load as an EncoderClassifier.  Either way the operator
    must not raise: if the model loads it classifies; if it fails to load it
    returns cuts unchanged (no-op).
    """
    from voxkitchen.operators.annotate.speechbrain_gender import (
        SpeechBrainGenderConfig,
        SpeechBrainGenderOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = SpeechBrainGenderConfig()
    op = SpeechBrainGenderOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cuts = list(result)

    assert len(out_cuts) == 1
    # Operator either added a supervision or returned cut unchanged — no crash either way
