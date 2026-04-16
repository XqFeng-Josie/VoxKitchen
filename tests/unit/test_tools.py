"""Unit tests for voxkitchen.tools standalone functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from voxkitchen.tools import estimate_snr, normalize_loudness, resample_audio


def _torchaudio_available() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def test_estimate_snr_returns_positive_float(mono_wav_16k: Path) -> None:
    snr = estimate_snr(mono_wav_16k)
    assert isinstance(snr, float)
    assert snr > 0


@pytest.mark.skipif(not _torchaudio_available(), reason="torchaudio not available")
def test_resample_audio_creates_output(mono_wav_16k: Path, tmp_path: Path) -> None:
    out = tmp_path / "resampled.wav"
    result = resample_audio(mono_wav_16k, out, target_sr=8000)
    assert result == out
    assert out.exists()
    import soundfile as sf

    info = sf.info(str(out))
    assert info.samplerate == 8000


def test_normalize_loudness_creates_output(mono_wav_16k: Path, tmp_path: Path) -> None:
    out = tmp_path / "normalized.wav"
    result = normalize_loudness(mono_wav_16k, out, target_lufs=-23.0)
    assert result == out
    assert out.exists()
