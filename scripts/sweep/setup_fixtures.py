"""Generate derived fixtures for the operator sweep.

Inputs that must already exist:
- ``examples/demo_data/demo1.opus`` (committed in repo)
- ``scripts/sweep/fixtures/audio/zh-tiny.wav`` (committed, sourced from THCHS-30)
- ``scripts/sweep/fixtures/text/reference.txt`` (committed)

Outputs (gitignored, regenerated each --setup):
- ``audio/tiny-english.wav`` — 5s slice of demo1.opus @ 16 kHz mono
- ``audio/demo1.opus`` — symlink to the canonical source
- ``noise/white-5s.wav`` — 5s of band-limited white noise
- ``rir/synthetic-rir.wav`` — 0.3s synthetic impulse response
- ``manifests/text-en-1cut.jsonl.gz`` — 1-cut text manifest for English TTS sweep
- ``manifests/text-zh-1cut.jsonl.gz`` — 1-cut text manifest for Chinese TTS sweep
"""

from __future__ import annotations

import gzip
import json
import subprocess
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def generate_fixtures(*, repo_root: Path, fixtures_dir: Path) -> None:
    """Generate all derived fixtures under ``fixtures_dir``.

    Idempotent: re-running produces byte-identical files (deterministic seed,
    fixed ffmpeg flags, fixed manifest content).
    """
    audio_dir = fixtures_dir / "audio"
    noise_dir = fixtures_dir / "noise"
    rir_dir = fixtures_dir / "rir"
    manifests_dir = fixtures_dir / "manifests"
    for d in (audio_dir, noise_dir, rir_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    _make_demo_symlink(repo_root, audio_dir)
    _make_tiny_english(repo_root, audio_dir)
    _make_white_noise(noise_dir)
    _make_synthetic_rir(rir_dir)
    _make_text_manifests(manifests_dir)


def _make_demo_symlink(repo_root: Path, audio_dir: Path) -> None:
    """Symlink (or copy on fs that can't symlink) demo1.opus into fixtures/audio."""
    src = repo_root / "examples" / "demo_data" / "demo1.opus"
    if not src.exists():
        raise FileNotFoundError(f"missing canonical source audio: {src} — fixtures cannot be built")
    dst = audio_dir / "demo1.opus"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:
        import shutil

        shutil.copy(src, dst)


def _make_tiny_english(repo_root: Path, audio_dir: Path) -> None:
    """Extract a deterministic 5s slice of demo1.opus → 16 kHz mono WAV.

    Slice offset is 19s (clear English in the demo); duration is 5s.
    """
    src = repo_root / "examples" / "demo_data" / "demo1.opus"
    dst = audio_dir / "tiny-english.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            "19",
            "-t",
            "5",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst),
        ],
        check=True,
    )


def _make_white_noise(noise_dir: Path) -> None:
    """5s of float32 white noise @ 16 kHz, seed=0 → deterministic."""
    dst = noise_dir / "white-5s.wav"
    rng = np.random.default_rng(seed=0)
    samples = rng.normal(0, 0.1, size=16000 * 5).astype(np.float32)
    _write_wav(dst, samples, sample_rate=16000)


def _make_synthetic_rir(rir_dir: Path) -> None:
    """0.3s synthetic impulse: spike + exponential decay. Deterministic."""
    dst = rir_dir / "synthetic-rir.wav"
    n = int(16000 * 0.3)
    t = np.arange(n) / 16000.0
    rir = np.exp(-t / 0.05).astype(np.float32)
    rir[0] = 1.0
    _write_wav(dst, rir, sample_rate=16000)


def _write_wav(path: Path, samples: np.ndarray, *, sample_rate: int) -> None:
    """Write float32 mono samples as 16-bit PCM WAV (deterministic across runs)."""
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _make_text_manifests(manifests_dir: Path) -> None:
    """Build two 1-cut text-only manifests for the TTS sweep ops."""
    _write_text_manifest(
        manifests_dir / "text-en-1cut.jsonl.gz",
        cut_id="sweep-en-1",
        text="VoxKitchen makes speech data preparation easier.",
    )
    _write_text_manifest(
        manifests_dir / "text-zh-1cut.jsonl.gz",
        cut_id="sweep-zh-1",
        text="今天天气不错，适合出门散步。",  # noqa: RUF001
    )


def _write_text_manifest(path: Path, *, cut_id: str, text: str) -> None:
    """Write a 1-cut text-only manifest in VoxKitchen's CutSet schema."""
    fixed_ts = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    header = {
        "__type__": "voxkitchen.header",
        "schema_version": "0.1",
        "created_at": fixed_ts,
        "pipeline_run_id": "sweep-fixture",
        "stage_name": "ingest",
    }
    cut = {
        "__type__": "cut",
        "id": cut_id,
        "recording_id": cut_id,
        "start": 0.0,
        "duration": 0.0,
        "channel": None,
        "recording": None,
        "supervisions": [
            {
                "id": "s0",
                "recording_id": cut_id,
                "start": 0.0,
                "duration": 0.0,
                "channel": None,
                "text": text,
                "language": None,
                "speaker": None,
                "gender": None,
                "age_range": None,
                "custom": {},
            }
        ],
        "metrics": {},
        "provenance": {
            "source_cut_id": None,
            "generated_by": "sweep-fixture",
            "stage_name": "ingest",
            "created_at": fixed_ts,
            "pipeline_run_id": "sweep-fixture",
        },
        "custom": {},
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(header, sort_keys=True) + "\n")
        f.write(json.dumps(cut, sort_keys=True) + "\n")
