"""Generate derived fixtures for the operator sweep.

Inputs that must already exist:
- ``examples/demo_data/demo1.opus`` (committed in repo)
- ``scripts/sweep/fixtures/audio/zh-tiny.wav`` (committed, sourced from THCHS-30)
- ``scripts/sweep/fixtures/text/reference.txt`` (committed)

Outputs (gitignored, regenerated each --setup):
- ``audio/tiny-english.wav`` — 5s slice of demo1.opus @ 16 kHz mono
- ``audio/demo1.opus`` — copy of the canonical source (was a symlink before; absolute host paths don't resolve inside containers)
- ``noise/white-5s.wav`` — 5s of band-limited white noise
- ``rir/synthetic-rir.wav`` — 0.3s synthetic impulse response
- ``manifests/text-en-1cut.jsonl.gz`` — 1-cut text manifest for English TTS sweep
- ``manifests/text-zh-1cut.jsonl.gz`` — 1-cut text manifest for Chinese TTS sweep
- ``manifests/cer-wer-1cut.jsonl.gz`` — 1 audio cut with supervisions.text + custom.reference_text
- ``manifests/text-markup-1cut.jsonl.gz`` — 1-cut text manifest with SenseVoice markup tags + double space for normalize_text sweep
- ``embeddings/ref-speaker.npy`` — 192-dim unit vector as reference speaker embedding
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import warnings
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

    embeddings_dir = fixtures_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    _make_demo_copy(repo_root, audio_dir)
    _make_tiny_english(repo_root, audio_dir)
    _make_white_noise(noise_dir)
    _make_synthetic_rir(rir_dir)
    _make_text_manifests(manifests_dir)
    _make_cer_wer_manifest(manifests_dir)
    _make_ref_speaker_embedding(embeddings_dir)
    _make_zh_subdir(repo_root, audio_dir)


def _make_demo_copy(repo_root: Path, audio_dir: Path) -> None:
    """Copy demo1.opus into fixtures/audio.

    A plain copy (not a symlink) is used so the file is accessible inside
    Docker containers, where the absolute host path the symlink would point to
    does not exist.  The fixture file is small (~460 KB) so the copy is cheap.
    """
    src = repo_root / "examples" / "demo_data" / "demo1.opus"
    if not src.exists():
        raise FileNotFoundError(f"missing canonical source audio: {src} — fixtures cannot be built")
    dst = audio_dir / "demo1.opus"
    if dst.is_symlink():
        # Replace any legacy symlink left by earlier setup runs with a real file.
        dst.unlink()
    if not dst.exists():
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
    _write_text_manifest(
        manifests_dir / "text-markup-1cut.jsonl.gz",
        cut_id="sweep-markup-1",
        # SenseVoice-style markup tags + a double space between "hello" and
        # "world" — so normalize_text has something meaningful to strip and the
        # assert_normalize_text_strips assertion (no <| and no double space) is
        # a real check rather than a vacuous pass.
        text="<|en|><|HAPPY|><|Speech|> hello  world <|withitn|>",
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
    # Use GzipFile with mtime=0 so the output bytes are deterministic — the
    # default gzip.open embeds the current wall-clock mtime in the header
    # bytes, which breaks the "byte-identical across runs" idempotency
    # contract in the module docstring.
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8") as f,
    ):
        f.write(json.dumps(header, sort_keys=True) + "\n")
        f.write(json.dumps(cut, sort_keys=True) + "\n")


def _make_cer_wer_manifest(manifests_dir: Path) -> None:
    """1-cut audio manifest for the cer_wer sweep.

    The cut references ``tiny-english.wav`` (a real audio file) and carries:
    - ``supervisions[0].text`` — a plausible ASR hypothesis
    - ``custom["reference_text"]`` — the ground-truth reference text

    Because cer_wer reads from ``cut.custom["reference_text"]`` (not from an
    upstream ASR stage), this fixture lets the operator run end-to-end in the
    slim image without any ASR model.
    """
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
        "id": "sweep-cerwer-1",
        "recording_id": "sweep-cerwer-1",
        "start": 0.0,
        "duration": 5.0,
        "channel": 0,
        "recording": {
            "id": "sweep-cerwer-1",
            "sampling_rate": 16000,
            "num_samples": 80000,
            "duration": 5.0,
            "num_channels": 1,  # Recording schema uses num_channels, not channel_ids
            "sources": [
                {
                    "type": "file",
                    "channels": [0],
                    "source": "/app/scripts/sweep/fixtures/audio/tiny-english.wav",
                }
            ],
        },
        "supervisions": [
            {
                "id": "s0",
                "recording_id": "sweep-cerwer-1",
                "start": 0.0,
                "duration": 5.0,
                "channel": 0,
                # Simulated ASR hypothesis — close to, but not identical to, reference.
                "text": "i don't know what it is",
                "language": "en",
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
        # reference_text is what cer_wer compares the ASR hypothesis against.
        "custom": {"reference_text": "I don't know what it is"},
    }
    path = manifests_dir / "cer-wer-1cut.jsonl.gz"
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        io.TextIOWrapper(gz, encoding="utf-8") as f,
    ):
        f.write(json.dumps(header, sort_keys=True) + "\n")
        f.write(json.dumps(cut, sort_keys=True) + "\n")


def _make_zh_subdir(repo_root: Path, audio_dir: Path) -> None:
    """Create fixtures/audio-zh/ with the committed zh-tiny.wav for Chinese ASR ops.

    Uses a relative symlink because audio-zh/ and audio/ are sibling directories
    inside the fixtures dir — the bind-mounted fixtures tree is internally
    consistent. Unlike _make_demo_copy (which copies because its source lives
    outside the bind-mounted fixtures dir), a relative symlink works here.

    ``zh-tiny.wav`` is a committed file in ``scripts/sweep/fixtures/audio/``.
    It is not a derived artifact so it may not be present in a fresh ``audio_dir``
    (e.g. a tmp path used by unit tests). We look first in ``audio_dir`` (for the
    in-place real-fixtures run) and fall back to the committed repo path.
    """
    src = audio_dir / "zh-tiny.wav"
    if not src.exists():
        # Fall back to committed repo fixture so unit tests don't need to
        # pre-copy the file into a tmp audio_dir.
        committed = repo_root / "scripts" / "sweep" / "fixtures" / "audio" / "zh-tiny.wav"
        if not committed.exists():
            raise FileNotFoundError(
                f"missing committed Chinese fixture: {committed} — Task 1 must commit this"
            )
        src = committed
    zh_dir = audio_dir.parent / "audio-zh"
    zh_dir.mkdir(parents=True, exist_ok=True)
    dst = zh_dir / "zh-tiny.wav"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    # Use a relative symlink (../audio/zh-tiny.wav) so the link resolves
    # correctly both on the host AND inside the Docker container where the
    # fixtures dir is bind-mounted at a different absolute path.
    try:
        rel_target = Path("..") / "audio" / "zh-tiny.wav"
        dst.symlink_to(rel_target)
    except OSError as exc:
        import shutil as _shutil

        warnings.warn(
            f"symlink_to({dst}) failed ({type(exc).__name__}: {exc}); falling back to shutil.copy",
            stacklevel=2,
        )
        _shutil.copy(src, dst)


def _make_ref_speaker_embedding(embeddings_dir: Path) -> None:
    """192-dim unit vector as a synthetic reference speaker embedding.

    The ECAPA-TDNN model (speechbrain/spkrec-ecapa-voxceleb) produces 192-dim
    embeddings. This fixture is a random unit vector with a fixed seed so the
    file is byte-identical across ``--setup`` runs. The speaker_similarity
    assertion only checks that the metric is in [-1, 1], so the exact values
    do not matter — cosine similarity between the unit vector and any real
    ECAPA embedding will always be in range.
    """
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(192).astype(np.float32)
    vec /= float(np.linalg.norm(vec))  # unit vector
    dst = embeddings_dir / "ref-speaker.npy"
    np.save(dst, vec)
