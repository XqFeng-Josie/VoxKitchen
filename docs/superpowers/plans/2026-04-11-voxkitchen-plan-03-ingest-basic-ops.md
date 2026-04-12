# VoxKitchen Plan 3: Ingest + Basic Operators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the directory-scan ingest source, 4 basic audio-processing operators (format conversion, resampling, channel merge, loudness normalization), and a manifest pack operator. This is the first plan that touches actual audio bytes — after it ships, a user can scan a folder of wav/mp3/flac files, resample them, normalize loudness, and produce a training-ready manifest.

**Architecture:** Audio operators follow a common pattern: read audio from the source file referenced by `cut.recording`, process it, write the result to `stage_dir/derived/`, and return a new Cut with a new embedded Recording pointing at the derived file. A shared `utils/audio.py` module provides `load_audio_for_cut()` and `save_audio()` to keep operators focused on their transform logic. The `recording` field is added to Cut as an optional embedded `Recording` — this follows Lhotse's self-contained Cut design and avoids the complexity of a separate `recordings.jsonl.gz` management layer.

**Tech Stack:** soundfile (audio I/O), ffmpeg-python (format conversion), torchaudio (resampling, channel ops), pyloudnorm (loudness normalization), numpy.

**Spec reference:** Design spec sections 2, 4.6 (Basic operator table), 5.

**Prior art:** Plan 2 (`plan-02-pipeline-engine` tag) delivered the full pipeline engine.

---

## File Structure Produced by This Plan

```
src/voxkitchen/
├── schema/
│   └── cut.py                      # MODIFIED: add recording: Recording | None = None
│
├── utils/
│   └── audio.py                    # NEW: load_audio_for_cut, save_audio, detect_audio_files
│
├── ingest/
│   ├── __init__.py                 # MODIFIED: register DirScanIngestSource
│   └── dir_scan.py                 # NEW: scan a directory of audio files → CutSet
│
├── operators/
│   ├── __init__.py                 # MODIFIED: register all new operators
│   ├── basic/
│   │   ├── __init__.py
│   │   ├── ffmpeg_convert.py       # NEW: format conversion via ffmpeg
│   │   ├── resample.py             # NEW: sample-rate conversion via torchaudio
│   │   ├── channel_merge.py        # NEW: stereo → mono (or N-channel → M-channel)
│   │   └── loudness_normalize.py   # NEW: LUFS-based loudness normalization
│   └── pack/
│       ├── __init__.py
│       └── pack_manifest.py        # NEW: write CutSet as-is → output manifest

tests/
├── unit/
│   ├── schema/
│   │   └── test_cut.py             # MODIFIED: add tests for recording field
│   ├── utils/
│   │   └── test_audio.py           # NEW
│   ├── ingest/
│   │   └── test_dir_scan.py        # NEW
│   ├── operators/
│   │   ├── basic/
│   │   │   ├── __init__.py
│   │   │   ├── test_ffmpeg_convert.py   # NEW
│   │   │   ├── test_resample.py         # NEW
│   │   │   ├── test_channel_merge.py    # NEW
│   │   │   └── test_loudness_normalize.py # NEW
│   │   └── pack/
│   │       ├── __init__.py
│   │       └── test_pack_manifest.py    # NEW
│
├── conftest.py                     # MODIFIED: add audio fixture generators
│
└── integration/
    └── test_audio_pipeline_e2e.py  # NEW: dir_scan → resample → normalize → pack
```

---

## Key Design Decision: Embedded Recording in Cut

Instead of managing a separate `recordings.jsonl.gz` file through the pipeline, each Cut carries an optional `recording: Recording | None = None` field. Audio-processing operators read from `cut.recording.sources[0].source` and create new Cuts with new embedded Recordings pointing at derived files.

**Why this approach:**
- Self-contained Cuts that can be serialized/deserialized independently
- No global mutable state to track across stages
- Matches the Lhotse MonoCut pattern (proven in production)
- Manifests are slightly larger but still JSONL.gz compressed (~negligible overhead)
- Operators stay simple: they read `cut.recording` and return new cuts with new recordings

**Backward compatibility:** The `recording` field is `Optional` with default `None`. All existing Plan 1/2 tests pass unchanged — their Cuts simply have `recording=None`.

---

## Task 1: Add audio dependencies and update CI

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Update `pyproject.toml` dependencies**

Add to the `dependencies` list (core deps):
```toml
"numpy>=1.24",
"ffmpeg-python>=0.2",
"pyloudnorm>=0.1",
```

Add a new extras group for torch (heavy, optional until Plan 8 merges to core):
```toml
[project.optional-dependencies]
audio = ["torch>=2.1", "torchaudio>=2.1"]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pre-commit>=3.6",
]
```

Note: `numpy` is already an implicit dep (pulled in by soundfile), but making it explicit pins the version. `soundfile>=0.12` is already in deps.

- [ ] **Step 2: Update CI to install audio extras with CPU-only torch**

In `.github/workflows/ci.yml`, update the Install step:

```yaml
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev]"
```

This installs CPU-only torch (~200MB, not the 2.5GB CUDA bundle) and then the project with dev extras. The `ffmpeg` binary is pre-installed on ubuntu-latest.

- [ ] **Step 3: Install locally**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Or if torch is already installed (e.g., CUDA version on a GPU machine): `pip install -e ".[dev]"` suffices since ffmpeg-python and pyloudnorm are light.

- [ ] **Step 4: Add mypy overrides for new deps**

Update `pyproject.toml` mypy overrides:
```toml
[[tool.mypy.overrides]]
module = ["soundfile.*", "ffmpeg.*", "pyloudnorm.*", "torch.*", "torchaudio.*"]
ignore_missing_imports = true
```

Merge the existing `soundfile.*` override into this one.

- [ ] **Step 5: Verify existing tests still pass**

```bash
pytest -q
```

Expected: all 121 tests pass.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml
git commit -m "chore: add audio deps (ffmpeg-python, pyloudnorm, torch extras)"
```

---

## Task 2: Add `recording` field to Cut + audio test fixtures

**Files:**
- Modify: `src/voxkitchen/schema/cut.py`
- Modify: `tests/conftest.py`
- Modify: `tests/unit/schema/test_cut.py`

- [ ] **Step 1: Add `recording` field to Cut**

Edit `src/voxkitchen/schema/cut.py` — add this field after `channel`:

```python
from voxkitchen.schema.recording import Recording

class Cut(BaseModel):
    ...
    channel: int | list[int] | None = None
    recording: Recording | None = None   # NEW: embedded for audio-processing operators
    supervisions: list[Supervision]
    ...
```

This is a backward-compatible addition: existing manifests without `recording` will deserialize with `recording=None`.

- [ ] **Step 2: Add tests for the new field**

Append to `tests/unit/schema/test_cut.py`:

```python
from voxkitchen.schema.recording import AudioSource, Recording

def test_cut_with_embedded_recording() -> None:
    rec = Recording(
        id="rec-1",
        sources=[AudioSource(type="file", channels=[0], source="/data/test.wav")],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
        num_channels=1,
    )
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        recording=rec,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut.recording is not None
    assert cut.recording.sampling_rate == 16000
    assert cut.recording.sources[0].source == "/data/test.wav"


def test_cut_recording_defaults_to_none() -> None:
    cut = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=_make_provenance(),
    )
    assert cut.recording is None


def test_cut_with_recording_round_trips_through_json() -> None:
    rec = Recording(
        id="rec-1",
        sources=[AudioSource(type="file", channels=[0], source="/data/test.wav")],
        sampling_rate=16000,
        num_samples=16000,
        duration=1.0,
        num_channels=1,
    )
    original = Cut(
        id="cut-1",
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        recording=rec,
        supervisions=[],
        provenance=_make_provenance(),
    )
    blob = original.model_dump_json()
    restored = Cut.model_validate_json(blob)
    assert restored.recording is not None
    assert restored.recording.id == "rec-1"
    assert restored == original
```

- [ ] **Step 3: Add audio fixture generators to `tests/conftest.py`**

Add these fixtures for generating real audio files:

```python
import numpy as np
import soundfile as sf

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
    import shutil
    shutil.copy(mono_wav_16k, audio_root / "mono.wav")
    shutil.copy(stereo_wav_44k, audio_root / "stereo.wav")
    # Also create a subdirectory with one more file
    sub = audio_root / "sub"
    sub.mkdir()
    sr = 16000
    t = np.linspace(0, 1, sr, dtype=np.float32)
    sf.write(sub / "deep.wav", 0.5 * np.sin(2 * np.pi * 660 * t), sr)
    return audio_root
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/unit/schema/test_cut.py -v
pytest -q  # full suite
```

Expected: all pass (121 + 3 new = 124).

- [ ] **Step 5: Commit**

```bash
git add src/voxkitchen/schema/cut.py tests/unit/schema/test_cut.py tests/conftest.py
git commit -m "feat(schema): add optional recording field to Cut + audio fixtures"
```

---

## Task 3: Audio utility module (TDD)

**Files:**
- Create: `tests/unit/utils/test_audio.py`
- Create: `src/voxkitchen/utils/audio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/utils/test_audio.py`:

```python
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
    assert len(files) == 3  # mono.wav, stereo.wav, sub/deep.wav
    # All are valid Paths
    assert all(f.exists() for f in files)


def test_detect_audio_files_non_recursive(audio_dir: Path) -> None:
    files = detect_audio_files(audio_dir, recursive=False)
    assert len(files) == 2  # only top-level files, not sub/deep.wav
```

- [ ] **Step 2: Run, see it fail**

```bash
pytest tests/unit/utils/test_audio.py -v
```

Expected: module not found.

- [ ] **Step 3: Write `src/voxkitchen/utils/audio.py`**

```python
"""Audio loading, saving, and file detection utilities.

These are thin wrappers around soundfile that standardize the interface
for VoxKitchen's audio-processing operators. Every operator that touches
audio bytes should use these helpers instead of calling soundfile directly.
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


def load_audio_for_cut(cut: Cut) -> tuple[np.ndarray, int]:
    """Load audio samples for a Cut from its embedded Recording.

    Returns (audio_float32, sample_rate). Audio is always float32.
    If the Cut covers a sub-interval of the Recording, only that interval
    is loaded.
    """
    if cut.recording is None:
        raise ValueError(f"cut {cut.id!r} has no embedded recording — cannot load audio")
    rec = cut.recording
    source_path = rec.sources[0].source
    audio, sr = sf.read(source_path, dtype="float32", start=0, stop=rec.num_samples)
    return audio, sr


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write audio samples to a file. Parent directories are created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def detect_audio_files(root: Path, *, recursive: bool = True) -> list[Path]:
    """Find audio files under a directory, sorted by name.

    Recognized extensions: .wav, .flac, .ogg, .mp3, .m4a, .opus, .wma, .aac
    """
    if recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()]
    return sorted(files)
```

- [ ] **Step 4: Run, see it pass**

```bash
pytest tests/unit/utils/test_audio.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/voxkitchen/utils/audio.py tests/unit/utils/test_audio.py
git commit -m "feat(utils): add audio load/save/detect utilities"
```

---

## Task 4: DirScanIngestSource (TDD)

**Files:**
- Create: `tests/unit/ingest/test_dir_scan.py`
- Create: `src/voxkitchen/ingest/dir_scan.py`
- Modify: `src/voxkitchen/ingest/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ingest/test_dir_scan.py`:

```python
"""Unit tests for voxkitchen.ingest.dir_scan.DirScanIngestSource."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.ingest.dir_scan import DirScanConfig, DirScanIngestSource
from voxkitchen.pipeline.context import RunContext


def _ctx(work_dir: Path) -> RunContext:
    return RunContext(
        work_dir=work_dir,
        pipeline_run_id="run-test",
        stage_index=0,
        stage_name="ingest",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def test_dir_scan_finds_all_audio_files(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=True),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 3  # mono.wav, stereo.wav, sub/deep.wav


def test_dir_scan_creates_cuts_with_embedded_recordings(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=True),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    for cut in cuts:
        assert cut.recording is not None
        assert cut.recording.sampling_rate > 0
        assert cut.recording.sources[0].type == "file"
        assert Path(cut.recording.sources[0].source).exists()
        assert cut.duration > 0
        assert cut.recording_id == cut.recording.id


def test_dir_scan_non_recursive(audio_dir: Path, tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(audio_dir), recursive=False),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 2  # only top-level


def test_dir_scan_rejects_nonexistent_directory(tmp_path: Path) -> None:
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(tmp_path / "nope")),
        ctx=_ctx(tmp_path),
    )
    with pytest.raises(FileNotFoundError):
        ingest.run()


def test_dir_scan_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty_dir"
    empty.mkdir()
    ingest = DirScanIngestSource(
        DirScanConfig(root=str(empty)),
        ctx=_ctx(tmp_path),
    )
    cuts = ingest.run()
    assert len(cuts) == 0
```

- [ ] **Step 2: Run, see it fail**

```bash
pytest tests/unit/ingest/test_dir_scan.py -v
```

- [ ] **Step 3: Write `src/voxkitchen/ingest/dir_scan.py`**

```python
"""DirScanIngestSource: scan a directory of audio files → CutSet.

Recursively (or non-recursively) discovers audio files, reads their
metadata via soundfile, and creates a Cut (with an embedded Recording)
for each one. This is the most common entry point for local data.
"""

from __future__ import annotations

from pathlib import Path

from voxkitchen.ingest.base import IngestConfig, IngestSource
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import detect_audio_files, recording_from_file
from voxkitchen.utils.time import now_utc


class DirScanConfig(IngestConfig):
    """Parameters for DirScanIngestSource."""

    root: str
    recursive: bool = True


class DirScanIngestSource(IngestSource):
    name = "dir"
    config_cls = DirScanConfig

    def run(self) -> CutSet:
        assert isinstance(self.config, DirScanConfig)
        root = Path(self.config.root)
        if not root.is_dir():
            raise FileNotFoundError(f"audio directory not found: {root}")

        audio_files = detect_audio_files(root, recursive=self.config.recursive)
        cuts: list[Cut] = []
        for audio_path in audio_files:
            rec = recording_from_file(audio_path)
            cut = Cut(
                id=rec.id,
                recording_id=rec.id,
                start=0.0,
                duration=rec.duration,
                recording=rec,
                supervisions=[],
                provenance=Provenance(
                    source_cut_id=None,
                    generated_by="dir_scan",
                    stage_name=self.ctx.stage_name,
                    created_at=now_utc(),
                    pipeline_run_id=self.ctx.pipeline_run_id,
                ),
            )
            cuts.append(cut)
        return CutSet(cuts)
```

- [ ] **Step 4: Register in `src/voxkitchen/ingest/__init__.py`**

Add `DirScanIngestSource` to imports and `_INGEST_SOURCES`:

```python
from voxkitchen.ingest.dir_scan import DirScanConfig, DirScanIngestSource

_INGEST_SOURCES: dict[str, type[IngestSource]] = {
    "dir": DirScanIngestSource,
    "manifest": ManifestIngestSource,
}
```

Add to `__all__`.

- [ ] **Step 5: Run, see it pass**

```bash
pytest tests/unit/ingest/test_dir_scan.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/ingest/dir_scan.py src/voxkitchen/ingest/__init__.py tests/unit/ingest/test_dir_scan.py
git commit -m "feat(ingest): add DirScanIngestSource for local audio directories"
```

---

## Task 5: ffmpeg_convert operator (TDD)

**Files:**
- Create: `src/voxkitchen/operators/basic/__init__.py`
- Create: `tests/unit/operators/basic/__init__.py`
- Create: `tests/unit/operators/basic/test_ffmpeg_convert.py`
- Create: `src/voxkitchen/operators/basic/ffmpeg_convert.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Create directories and init files**

```bash
mkdir -p src/voxkitchen/operators/basic
mkdir -p tests/unit/operators/basic
```

Write empty `__init__.py` in both.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/operators/basic/test_ffmpeg_convert.py`:

```python
"""Unit tests for voxkitchen.operators.basic.ffmpeg_convert."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import soundfile as sf

from voxkitchen.operators.basic.ffmpeg_convert import FfmpegConvertConfig, FfmpegConvertOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import AudioSource, Recording
from voxkitchen.utils.audio import recording_from_file


def _ctx(tmp_path: Path) -> RunContext:
    return RunContext(
        work_dir=tmp_path,
        pipeline_run_id="run-test",
        stage_index=1,
        stage_name="convert",
        num_gpus=0,
        num_cpu_workers=1,
        gc_mode="aggressive",
        device="cpu",
    )


def _cut_from_path(audio_path: Path) -> Cut:
    from datetime import datetime, timezone

    rec = recording_from_file(audio_path)
    return Cut(
        id=f"cut-{rec.id}",
        recording_id=rec.id,
        start=0.0,
        duration=rec.duration,
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


def test_ffmpeg_convert_is_registered() -> None:
    assert get_operator("ffmpeg_convert") is FfmpegConvertOperator


def test_ffmpeg_convert_produces_audio() -> None:
    assert FfmpegConvertOperator.produces_audio is True


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_convert_wav_to_flac(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = FfmpegConvertConfig(target_format="flac")
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    assert len(result) == 1
    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.suffix == ".flac"
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.samplerate == 16000


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_convert_preserves_cut_metadata(mono_wav_16k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    original_cut = _cut_from_path(mono_wav_16k)
    cs = CutSet([original_cut])
    config = FfmpegConvertConfig(target_format="wav")
    op = FfmpegConvertOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.provenance.source_cut_id == original_cut.id
    assert out_cut.provenance.generated_by.startswith("ffmpeg_convert")
```

- [ ] **Step 3: Write `src/voxkitchen/operators/basic/ffmpeg_convert.py`**

```python
"""Format conversion operator using ffmpeg.

Converts audio files to a target format (e.g., mp3 → wav, flac → wav).
This is a ``produces_audio=True`` operator: it writes new files to
``stage_dir/derived/``.

Requires the ``ffmpeg`` binary to be available on PATH.
"""

from __future__ import annotations

from pathlib import Path

import ffmpeg

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import recording_from_file
from voxkitchen.utils.time import now_utc


class FfmpegConvertConfig(OperatorConfig):
    target_format: str = "wav"


@register_operator
class FfmpegConvertOperator(Operator):
    name = "ffmpeg_convert"
    config_cls = FfmpegConvertConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, FfmpegConvertConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            if cut.recording is None:
                raise ValueError(f"cut {cut.id!r} has no recording")
            src = cut.recording.sources[0].source
            fmt = self.config.target_format
            out_name = f"{cut.id}.{fmt}"
            out_path = derived_dir / out_name

            (
                ffmpeg.input(src)
                .output(str(out_path), y=None)
                .overwrite_output()
                .run(quiet=True)
            )

            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording.id}_{fmt}")
            out_cuts.append(
                Cut(
                    id=f"{cut.id}__{fmt}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"ffmpeg_convert@{fmt}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return CutSet(out_cuts)
```

- [ ] **Step 4: Register in `operators/__init__.py`**

Add to the import block that triggers registration:

```python
from voxkitchen.operators.basic import ffmpeg_convert as _basic_ffmpeg  # noqa: F401
```

- [ ] **Step 5: Run, see it pass**

```bash
pytest tests/unit/operators/basic/test_ffmpeg_convert.py -v
```

Expected: 4 passed (or 2 skipped + 2 passed if ffmpeg isn't installed).

- [ ] **Step 6: Commit**

```bash
git add src/voxkitchen/operators/basic src/voxkitchen/operators/__init__.py tests/unit/operators/basic
git commit -m "feat(operators): add ffmpeg_convert for audio format conversion"
```

---

## Task 6: resample operator (TDD)

**Files:**
- Create: `tests/unit/operators/basic/test_resample.py`
- Create: `src/voxkitchen/operators/basic/resample.py`
- Modify: `src/voxkitchen/operators/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/operators/basic/test_resample.py`:

```python
"""Unit tests for voxkitchen.operators.basic.resample."""

from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from voxkitchen.operators.basic.resample import ResampleConfig, ResampleOperator
from voxkitchen.operators.registry import get_operator
from voxkitchen.pipeline.context import RunContext
from voxkitchen.schema.cutset import CutSet
from voxkitchen.utils.audio import recording_from_file

# Import _cut_from_path pattern — reuse from ffmpeg test or define locally
from tests.unit.operators.basic.test_ffmpeg_convert import _cut_from_path, _ctx


def test_resample_is_registered() -> None:
    assert get_operator("resample") is ResampleOperator


def test_resample_produces_audio() -> None:
    assert ResampleOperator.produces_audio is True


def test_resample_44k_to_16k(stereo_wav_44k: Path, tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(stereo_wav_44k)])
    config = ResampleConfig(target_sr=16000)
    op = ResampleOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    derived_path = Path(out_cut.recording.sources[0].source)
    assert derived_path.exists()
    info = sf.info(str(derived_path))
    assert info.samplerate == 16000


def test_resample_same_rate_is_noop(mono_wav_16k: Path, tmp_path: Path) -> None:
    """When target_sr matches the input, copy or skip."""
    ctx = _ctx(tmp_path)
    cs = CutSet([_cut_from_path(mono_wav_16k)])
    config = ResampleConfig(target_sr=16000)
    op = ResampleOperator(config, ctx)
    op.setup()
    result = op.process(cs)
    op.teardown()

    out_cut = next(iter(result))
    assert out_cut.recording is not None
    assert out_cut.recording.sampling_rate == 16000
```

**Note:** If importing `_cut_from_path` from another test file causes issues (pytest may not allow it cleanly), just duplicate the helper locally. The implementer should use their judgment.

- [ ] **Step 2: Implement `src/voxkitchen/operators/basic/resample.py`**

Uses `torchaudio.transforms.Resample` for high-quality resampling. Falls back to `soundfile` read/write if torch is not available.

```python
"""Resample operator: change the sample rate of audio files.

Uses torchaudio for high-quality resampling. Creates new audio files
in ``stage_dir/derived/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from voxkitchen.operators.base import Operator, OperatorConfig
from voxkitchen.operators.registry import register_operator
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.provenance import Provenance
from voxkitchen.utils.audio import load_audio_for_cut, recording_from_file, save_audio
from voxkitchen.utils.time import now_utc


class ResampleConfig(OperatorConfig):
    target_sr: int
    target_channels: int | None = None


@register_operator
class ResampleOperator(Operator):
    name = "resample"
    config_cls = ResampleConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
    required_extras = ["audio"]

    def process(self, cuts: CutSet) -> CutSet:
        assert isinstance(self.config, ResampleConfig)
        derived_dir = self.ctx.stage_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)

        out_cuts: list[Cut] = []
        for cut in cuts:
            audio, sr = load_audio_for_cut(cut)
            target_sr = self.config.target_sr

            if sr != target_sr:
                audio = self._resample(audio, sr, target_sr)
                sr = target_sr

            if self.config.target_channels is not None:
                audio = self._adjust_channels(audio, self.config.target_channels)

            out_path = derived_dir / f"{cut.id}.wav"
            save_audio(out_path, audio, sr)
            new_rec = recording_from_file(out_path, recording_id=f"{cut.recording_id}_rs{sr}")

            out_cuts.append(
                Cut(
                    id=f"{cut.id}__rs{sr}",
                    recording_id=new_rec.id,
                    start=0.0,
                    duration=new_rec.duration,
                    recording=new_rec,
                    supervisions=cut.supervisions,
                    metrics=cut.metrics,
                    provenance=Provenance(
                        source_cut_id=cut.id,
                        generated_by=f"resample@{sr}",
                        stage_name=self.ctx.stage_name,
                        created_at=now_utc(),
                        pipeline_run_id=self.ctx.pipeline_run_id,
                    ),
                    custom=cut.custom,
                )
            )
        return CutSet(out_cuts)

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        try:
            import torch
            import torchaudio

            if audio.ndim == 1:
                tensor = torch.from_numpy(audio).unsqueeze(0)
            else:
                tensor = torch.from_numpy(audio.T)
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
            resampled = resampler(tensor)
            if resampled.shape[0] == 1:
                return resampled.squeeze(0).numpy()
            return resampled.T.numpy()
        except ImportError:
            from scipy.signal import resample as scipy_resample

            new_len = int(len(audio) * target_sr / orig_sr)
            return scipy_resample(audio, new_len).astype(np.float32)

    @staticmethod
    def _adjust_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
        if audio.ndim == 1 and target_channels == 1:
            return audio
        if audio.ndim == 2 and target_channels == 1:
            return audio.mean(axis=1).astype(np.float32)
        if audio.ndim == 1 and target_channels > 1:
            return np.column_stack([audio] * target_channels)
        return audio
```

- [ ] **Step 3: Register + run tests + commit**

Register in `operators/__init__.py`. Run tests. Commit: `feat(operators): add resample operator with torchaudio`

---

## Task 7: channel_merge + loudness_normalize operators (TDD)

**Files:**
- Create: `tests/unit/operators/basic/test_channel_merge.py`
- Create: `src/voxkitchen/operators/basic/channel_merge.py`
- Create: `tests/unit/operators/basic/test_loudness_normalize.py`
- Create: `src/voxkitchen/operators/basic/loudness_normalize.py`
- Modify: `src/voxkitchen/operators/__init__.py`

### channel_merge

Converts stereo → mono (or N-channel → M-channel). Very similar to resample but focuses only on channel dimension.

```python
class ChannelMergeConfig(OperatorConfig):
    target_channels: int = 1

@register_operator
class ChannelMergeOperator(Operator):
    name = "channel_merge"
    config_cls = ChannelMergeConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
```

Process: load audio, merge channels (mean for mono, duplicate for expand), write new file. Same `derived/` + `recording_from_file` + new Cut pattern.

### loudness_normalize

LUFS-based loudness normalization using `pyloudnorm`.

```python
class LoudnessNormalizeConfig(OperatorConfig):
    target_lufs: float = -23.0

@register_operator
class LoudnessNormalizeOperator(Operator):
    name = "loudness_normalize"
    config_cls = LoudnessNormalizeConfig
    device = "cpu"
    produces_audio = True
    reads_audio_bytes = True
```

Process: load audio, measure loudness via `pyloudnorm.Meter`, normalize to target LUFS, clip to [-1, 1], write new file. Same derived pattern.

### Tests for each (3-4 tests each):
- Registered correctly
- `produces_audio` is True
- Process a stereo file → mono output (channel_merge)
- Process a quiet file → louder output with target LUFS (loudness_normalize)
- Provenance chain preserved

### Two commits: one for channel_merge, one for loudness_normalize

---

## Task 8: pack_manifest operator (TDD)

**Files:**
- Create: `src/voxkitchen/operators/pack/__init__.py`
- Create: `tests/unit/operators/pack/__init__.py`
- Create: `tests/unit/operators/pack/test_pack_manifest.py`
- Create: `src/voxkitchen/operators/pack/pack_manifest.py`
- Modify: `src/voxkitchen/operators/__init__.py`

The simplest "pack" operator: writes the CutSet as-is to a final manifest location. No audio transformation — just serializes the current CutSet to an output path specified in config.

```python
class PackManifestConfig(OperatorConfig):
    output_dir: str | None = None  # if None, use stage_dir

@register_operator
class PackManifestOperator(Operator):
    name = "pack_manifest"
    config_cls = PackManifestConfig
    device = "cpu"
    produces_audio = False
    reads_audio_bytes = False
```

### Tests (3):
- Registered correctly
- Process returns the same CutSet unchanged
- `produces_audio` is False, `reads_audio_bytes` is False

### Commit: `feat(operators): add pack_manifest output operator`

---

## Task 9: Integration test — real audio pipeline

**Files:**
- Create: `tests/integration/test_audio_pipeline_e2e.py`

### Test

A full end-to-end test using DirScan + resample + pack_manifest on real (generated) audio files.

```python
def test_dir_scan_to_resample_to_pack(audio_dir: Path, tmp_path: Path) -> None:
    """Full pipeline: scan dir → resample 44k→16k → pack manifest."""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(f"""
version: "0.1"
name: audio-e2e
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: resample
    op: resample
    args: {{ target_sr: 16000, target_channels: 1 }}
  - name: pack
    op: pack_manifest
""")
    spec = load_pipeline_spec(yaml_path, run_id="run-audio-e2e")
    run_pipeline(spec)

    # Verify resample stage produced derived audio
    resample_dir = work_dir / "00_resample"
    assert (resample_dir / "_SUCCESS").exists()
    derived = resample_dir / "derived"
    assert derived.exists()
    derived_files = list(derived.glob("*.wav"))
    assert len(derived_files) == 3  # one per input file

    # Verify all resampled files are 16kHz mono
    for f in derived_files:
        info = sf.info(str(f))
        assert info.samplerate == 16000

    # Verify pack stage completed
    assert (work_dir / "01_pack" / "_SUCCESS").exists()
    final_cuts = list(read_cuts(work_dir / "01_pack" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 3
```

### Commit: `test(integration): add real audio pipeline test (dir_scan → resample → pack)`

---

## Task 10: Full verification, lint, type, tag

- [ ] **Run the entire suite**: all Plan 1 + Plan 2 + Plan 3 tests
- [ ] **ruff check + format**: clean
- [ ] **mypy strict**: clean
- [ ] **pre-commit run --all-files**: all hooks pass
- [ ] **Tag**: `git tag -a plan-03-ingest-basic-ops -m "Plan 3 complete: DirScan ingest + basic audio operators"`

---

## Plan 3 Completion Checklist

- [ ] `pip install -e ".[dev]"` succeeds; `pip install torch torchaudio` adds audio capabilities
- [ ] `DirScanIngestSource` scans a directory and produces Cuts with embedded Recordings
- [ ] `ffmpeg_convert` converts between audio formats (wav/flac/mp3) — tested with real ffmpeg
- [ ] `resample` changes sample rate using torchaudio (falls back to scipy if torch unavailable)
- [ ] `channel_merge` converts stereo → mono
- [ ] `loudness_normalize` normalizes to target LUFS using pyloudnorm
- [ ] `pack_manifest` writes the CutSet through as-is (trivial passthrough)
- [ ] All `produces_audio=True` operators write to `stage_dir/derived/`
- [ ] All operators create proper provenance chains (source_cut_id + generated_by)
- [ ] Integration test runs a full pipeline on generated audio files
- [ ] GC would clean up `derived/` dirs after their consumers finish (tested via Plan 2's GC tests + the integration test)
- [ ] `vkit validate` still works on pipelines using the new operators
- [ ] All existing Plan 1 + Plan 2 tests still pass (no regressions)
- [ ] `git tag plan-03-ingest-basic-ops` placed at HEAD

## What Plan 4 Will Build On

Plan 4 adds segmentation + quality + pack operators:
- `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` (segmentation)
- `snr_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter` (quality)
- `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` (pack formats)

These operators consume the `recording` field and the `produces_audio=True` → `derived/` → GC infrastructure that Plan 3 established. None of them require changes to the runner, executor, or schema.
