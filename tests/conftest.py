"""Shared pytest fixtures for voxkitchen tests."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from voxkitchen.pipeline.context import RunContext


@pytest.fixture
def fixed_datetime() -> datetime:
    """A deterministic UTC datetime for reproducible tests."""
    return datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def make_run_context(tmp_path: Path) -> Callable[..., RunContext]:
    """Factory for a CPU-only RunContext parented at ``tmp_path``.

    Tests needing non-default process settings (``num_gpus``,
    ``num_cpu_workers``, ``device``, a specific ``pipeline_run_id``)
    should build their own ``RunContext`` — this factory intentionally
    covers only the common "run one operator against a fixture" case.
    """

    def _factory(
        stage_name: str = "test",
        *,
        stage_index: int = 1,
        pipeline_run_id: str = "run-test",
    ) -> RunContext:
        return RunContext(
            work_dir=tmp_path,
            pipeline_run_id=pipeline_run_id,
            stage_index=stage_index,
            stage_name=stage_name,
            num_gpus=0,
            num_cpu_workers=1,
            gc_mode="aggressive",
            device="cpu",
        )

    return _factory


@pytest.fixture
def tmp_jsonl_gz(tmp_path: Path) -> Path:
    """Return a path to a temporary .jsonl.gz file inside tmp_path."""
    return tmp_path / "cuts.jsonl.gz"


@pytest.fixture
def mono_wav_16k(tmp_path: Path) -> Path:
    """Generate a 1-second 16kHz mono sine wave."""
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "mono_16k.wav"
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def stereo_wav_44k(tmp_path: Path) -> Path:
    """Generate a 1-second 44.1kHz stereo sine wave."""
    sr = 44100
    t = np.linspace(0, 1, sr, dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.3 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack([left, right])
    path = tmp_path / "stereo_44k.wav"
    sf.write(path, audio, sr)
    return path


@pytest.fixture
def audio_dir(tmp_path: Path, mono_wav_16k: Path, stereo_wav_44k: Path) -> Path:
    """A directory containing a few audio files for DirScan tests."""
    audio_root = tmp_path / "audio_input"
    audio_root.mkdir()
    shutil.copy(mono_wav_16k, audio_root / "mono.wav")
    shutil.copy(stereo_wav_44k, audio_root / "stereo.wav")
    sub = audio_root / "sub"
    sub.mkdir()
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    sf.write(sub / "deep.wav", 0.5 * np.sin(2 * np.pi * 660 * t), sr)
    return audio_root
