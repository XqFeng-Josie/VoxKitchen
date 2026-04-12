"""Unit tests for audio_fingerprint_dedup operator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from voxkitchen.operators.registry import get_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording


def _make_cut(path: Path, cut_id: str | None = None) -> Cut:
    """Build a Cut from an audio file path."""
    info = sf.info(str(path))
    stem = cut_id or path.stem
    rec = Recording(
        id=f"rec-{stem}",
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=info.samplerate,
        num_samples=info.frames,
        duration=info.duration,
        num_channels=info.channels,
    )
    return Cut(
        id=stem,
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


def test_dedup_is_registered() -> None:
    from voxkitchen.operators.quality.audio_fingerprint_dedup import AudioFingerprintDedupOperator

    assert get_operator("audio_fingerprint_dedup") is AudioFingerprintDedupOperator


def test_dedup_removes_identical_cuts(mono_wav_16k: Path) -> None:
    """Two cuts pointing at the same audio file → only 1 kept."""
    from voxkitchen.operators.quality.audio_fingerprint_dedup import (
        AudioFingerprintDedupConfig,
        AudioFingerprintDedupOperator,
    )

    cut1 = _make_cut(mono_wav_16k, cut_id="cut-a")
    cut2 = _make_cut(mono_wav_16k, cut_id="cut-b")
    config = AudioFingerprintDedupConfig()
    op = AudioFingerprintDedupOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut1, cut2])))

    assert len(result) == 1


def test_dedup_keeps_different_cuts(mono_wav_16k: Path, stereo_wav_44k: Path) -> None:
    """Two different audio files → both kept."""
    from voxkitchen.operators.quality.audio_fingerprint_dedup import (
        AudioFingerprintDedupConfig,
        AudioFingerprintDedupOperator,
    )

    cut1 = _make_cut(mono_wav_16k, cut_id="cut-mono")
    cut2 = _make_cut(stereo_wav_44k, cut_id="cut-stereo")
    config = AudioFingerprintDedupConfig()
    op = AudioFingerprintDedupOperator(config, ctx=object())  # type: ignore[arg-type]
    result = list(op.process(CutSet([cut1, cut2])))

    assert len(result) == 2
