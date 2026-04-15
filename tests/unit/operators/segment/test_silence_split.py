"""Unit tests for silence_split operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("librosa")

from voxkitchen.operators.registry import get_operator  # noqa: E402
from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.schema.recording import AudioSource, Recording  # noqa: E402


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


def test_silence_split_is_registered() -> None:
    from voxkitchen.operators.segment.silence_split import SilenceSplitOperator

    assert get_operator("silence_split") is SilenceSplitOperator


def test_silence_split_on_continuous_tone(mono_wav_16k: Path) -> None:
    """A continuous sine wave has no silence → should produce ~1 segment covering full duration."""
    from voxkitchen.operators.segment.silence_split import SilenceSplitConfig, SilenceSplitOperator

    cut = _make_cut(mono_wav_16k)
    config = SilenceSplitConfig(top_db=30, min_duration=0.1)
    op = SilenceSplitOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))

    # Continuous tone is one non-silent region
    assert len(result) == 1
    # The single segment should cover most of the original duration
    assert result[0].duration == pytest.approx(cut.duration, abs=0.1)


def test_silence_split_on_silence(tmp_path: Path) -> None:
    """A fully silent file should produce 0 segments."""
    from voxkitchen.operators.segment.silence_split import SilenceSplitConfig, SilenceSplitOperator

    silence = np.zeros(16000, dtype=np.float32)
    silent_path = tmp_path / "silent.wav"
    sf.write(str(silent_path), silence, 16000)

    cut = _make_cut(silent_path)
    config = SilenceSplitConfig(top_db=30, min_duration=0.1)
    op = SilenceSplitOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))

    assert len(result) == 0
