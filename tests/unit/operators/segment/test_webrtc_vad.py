"""Unit tests for webrtc_vad operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import soundfile as sf

pytest.importorskip("webrtcvad")

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


def test_webrtc_vad_is_registered() -> None:
    from voxkitchen.operators.segment.webrtc_vad import WebrtcVadOperator

    assert get_operator("webrtc_vad") is WebrtcVadOperator


def test_webrtc_vad_detects_speech(mono_wav_16k: Path) -> None:
    """A sine wave should be detected as speech-like, producing >= 1 segment."""
    from voxkitchen.operators.segment.webrtc_vad import WebrtcVadConfig, WebrtcVadOperator

    cut = _make_cut(mono_wav_16k)
    config = WebrtcVadConfig(
        aggressiveness=2, frame_duration_ms=30, min_speech_duration_ms=100, padding_ms=30
    )
    op = WebrtcVadOperator(config, ctx=object())  # type: ignore[arg-type]
    op.setup()
    result = list(op.process(CutSet([cut])))

    assert len(result) >= 1


def test_webrtc_vad_produces_child_cuts_with_provenance(mono_wav_16k: Path) -> None:
    """Child cuts must carry source_cut_id pointing back to the parent cut."""
    from voxkitchen.operators.segment.webrtc_vad import WebrtcVadConfig, WebrtcVadOperator

    cut = _make_cut(mono_wav_16k)
    config = WebrtcVadConfig(
        aggressiveness=2, frame_duration_ms=30, min_speech_duration_ms=100, padding_ms=30
    )
    op = WebrtcVadOperator(config, ctx=object())  # type: ignore[arg-type]
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
