"""Unit tests for voxkitchen.utils.audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.utils.audio import (
    detect_audio_files,
    load_audio_for_cut,
    recording_from_file,
    save_audio,
)


def _make_recording(path: Path, sr: int = 16000, n_samples: int = 16000) -> Recording:
    return Recording(
        id=path.stem,
        sources=[AudioSource(type="file", channels=[0], source=str(path))],
        sampling_rate=sr,
        num_samples=n_samples,
        duration=n_samples / sr,
        num_channels=1,
    )


def _make_cut_with_recording(rec: Recording) -> Cut:
    from datetime import datetime, timezone

    return Cut(
        id=f"cut-{rec.id}",
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
        recording=rec,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="test",
            stage_name="test",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="test-run",
        ),
    )


def test_recording_from_file_reads_metadata(mono_wav_16k: Path) -> None:
    rec = recording_from_file(mono_wav_16k)
    assert rec.sampling_rate == 16000
    assert rec.num_channels == 1
    assert rec.num_samples == 16000
    assert abs(rec.duration - 1.0) < 0.01
    assert rec.sources[0].source == str(mono_wav_16k)
    assert rec.sources[0].type == "file"


def test_recording_from_file_stereo(stereo_wav_44k: Path) -> None:
    rec = recording_from_file(stereo_wav_44k)
    assert rec.sampling_rate == 44100
    assert rec.num_channels == 2


def test_load_audio_for_cut(mono_wav_16k: Path) -> None:
    rec = _make_recording(mono_wav_16k)
    cut = _make_cut_with_recording(rec)
    audio, sr = load_audio_for_cut(cut)
    assert sr == 16000
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 16000


def test_save_audio_roundtrips(tmp_path: Path) -> None:
    sr = 16000
    audio = np.random.randn(sr).astype(np.float32) * 0.5
    out = tmp_path / "out.wav"
    save_audio(out, audio, sr)
    loaded, loaded_sr = sf.read(out, dtype="float32")
    assert loaded_sr == sr
    assert np.allclose(audio, loaded, atol=1e-4)


def test_detect_audio_files_finds_wavs(audio_dir: Path) -> None:
    files = detect_audio_files(audio_dir, recursive=True)
    assert len(files) == 3
    assert all(f.exists() for f in files)


def test_detect_audio_files_non_recursive(audio_dir: Path) -> None:
    files = detect_audio_files(audio_dir, recursive=False)
    assert len(files) == 2
