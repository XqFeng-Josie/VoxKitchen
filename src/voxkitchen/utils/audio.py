"""Audio loading, saving, and file detection utilities.

Thin wrappers around soundfile that standardize the interface for
VoxKitchen's audio-processing operators.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from voxkitchen.schema.cut import Cut
from voxkitchen.schema.recording import AudioSource, Recording

AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".wma", ".aac"}


def recording_from_file(path: Path, recording_id: str | None = None) -> Recording:
    """Read audio metadata from a file and return a Recording."""
    info = sf.info(str(path))
    rid = recording_id or path.stem
    return Recording(
        id=rid,
        sources=[AudioSource(type="file", channels=list(range(info.channels)), source=str(path))],
        sampling_rate=info.samplerate,
        num_samples=info.frames,
        duration=info.duration,
        num_channels=info.channels,
    )


def load_audio_for_cut(cut: Cut) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float32]], int]:
    """Load audio samples for a Cut from its embedded Recording.

    Returns (audio_float32, sample_rate).
    """
    if cut.recording is None:
        raise ValueError(f"cut {cut.id!r} has no embedded recording")
    source_path = cut.recording.sources[0].source
    audio, sr = sf.read(source_path, dtype="float32")
    return audio, int(sr)


def save_audio(
    path: Path, audio: np.ndarray[tuple[int, ...], np.dtype[np.float32]], sample_rate: int
) -> None:
    """Write audio samples to a file. Parent dirs created automatically.

    WAV files are written as 32-bit float to preserve sample precision.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16"
    if path.suffix.lower() == ".wav":
        subtype = "FLOAT"
    sf.write(str(path), audio, sample_rate, subtype=subtype)


def detect_audio_files(root: Path, *, recursive: bool = True) -> list[Path]:
    """Find audio files under a directory, sorted by name."""
    if recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    return sorted(files)
