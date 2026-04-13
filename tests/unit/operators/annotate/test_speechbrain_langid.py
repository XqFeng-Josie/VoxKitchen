"""Unit tests for speechbrain_langid operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Skip entire module if speechbrain is not installed.
pytest.importorskip("speechbrain")

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
# Fast (no model download)
# ---------------------------------------------------------------------------


def test_speechbrain_langid_is_registered() -> None:
    from voxkitchen.operators.annotate.speechbrain_langid import SpeechBrainLangIdOperator

    assert get_operator("speechbrain_langid") is SpeechBrainLangIdOperator


def test_speechbrain_langid_class_attrs() -> None:
    from voxkitchen.operators.annotate.speechbrain_langid import SpeechBrainLangIdOperator

    assert SpeechBrainLangIdOperator.device == "gpu"
    assert SpeechBrainLangIdOperator.produces_audio is False
    assert SpeechBrainLangIdOperator.reads_audio_bytes is True
    assert "classify" in SpeechBrainLangIdOperator.required_extras


# ---------------------------------------------------------------------------
# Slow (downloads VoxLingua107 ECAPA model ~80 MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_speechbrain_langid_classifies(mono_wav_16k: Path) -> None:
    """Real model on CPU. Language of a sine wave is arbitrary but must be a string."""
    from voxkitchen.operators.annotate.speechbrain_langid import (
        SpeechBrainLangIdConfig,
        SpeechBrainLangIdOperator,
    )

    cut = _make_cut(mono_wav_16k)
    config = SpeechBrainLangIdConfig()
    op = SpeechBrainLangIdOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = op.process(CutSet([cut]))
    out_cuts = list(result)

    assert len(out_cuts) == 1
    out_cut = out_cuts[0]
    # Exactly one langid supervision added
    langid_sups = [s for s in out_cut.supervisions if s.id.endswith("__langid")]
    assert len(langid_sups) == 1
    lang = langid_sups[0].language
    assert isinstance(lang, str)
    assert len(lang) > 0
