"""Unit tests for pyannote_diarize operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip entire module if pyannote.audio is not installed.
pytest.importorskip("pyannote.audio")

from voxkitchen.operators.registry import get_operator  # noqa: E402
from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.schema.recording import AudioSource, Recording  # noqa: E402

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
# Fast (no model download — only checks import and registration)
# ---------------------------------------------------------------------------


def test_pyannote_diarize_is_registered() -> None:
    from voxkitchen.operators.annotate.pyannote_diarize import PyannoteDiarizeOperator

    assert get_operator("pyannote_diarize") is PyannoteDiarizeOperator


def test_pyannote_diarize_class_attrs() -> None:
    from voxkitchen.operators.annotate.pyannote_diarize import PyannoteDiarizeOperator

    assert PyannoteDiarizeOperator.device == "gpu"
    assert PyannoteDiarizeOperator.produces_audio is False


# ---------------------------------------------------------------------------
# GPU-only (requires HF_TOKEN + pyannote model access; skip on dev)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_pyannote_diarize_adds_speakers(mono_wav_16k: Path) -> None:
    """Requires HF_TOKEN + pyannote model access. Run on configured server only."""
    from voxkitchen.operators.annotate.pyannote_diarize import (
        PyannoteDiarizeConfig,
        PyannoteDiarizeOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = PyannoteDiarizeConfig()
    op = PyannoteDiarizeOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cuts = list(result)

    assert len(out_cuts) == 1
    for sup in out_cuts[0].supervisions:
        assert sup.speaker is not None
        assert sup.duration > 0
