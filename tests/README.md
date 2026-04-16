# Testing Guide

## Quick Reference

```bash
# Fast tests (no model download, seconds)
pytest tests/unit/operators/ -v -m "not slow and not gpu"

# All CPU tests (downloads models, minutes)
pytest tests/unit/operators/ -v -m "not gpu"

# GPU tests (needs CUDA + HF_TOKEN for pyannote)
HF_TOKEN=hf_xxx pytest tests/unit/operators/ -v -m "gpu"

# Everything
HF_TOKEN=hf_xxx pytest tests/unit/operators/ -v

# Single category
pytest tests/unit/operators/annotate/ -v
pytest tests/unit/operators/quality/ -v
pytest tests/unit/operators/synthesize/ -v

# Single operator
pytest tests/unit/operators/annotate/test_faster_whisper_asr.py -v

# Full test suite (all modules, not just operators)
pytest -v -m "not slow and not gpu"
```

## Docker (Recommended)

Docker provides a clean environment with all 51 operators and system deps pre-installed.

```bash
# Build (first time ~10-15 min, cached after)
docker build -t voxkitchen .

# All CPU tests (including slow model downloads)
docker run --rm --entrypoint pytest voxkitchen tests/unit/operators/ -v -m "not gpu"

# Fast tests only
docker run --rm --entrypoint pytest voxkitchen tests/unit/operators/ -v -m "not slow and not gpu"

# GPU tests
docker run --rm --gpus all --entrypoint pytest voxkitchen tests/unit/operators/ -v

# Interactive shell
docker run --rm -it --entrypoint bash voxkitchen
```

## Local Setup

```bash
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen

# GPU machine: install torch first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# System deps (needed by deepfilternet, av, etc.)
sudo apt-get install -y pkg-config ffmpeg \
    libavformat-dev libavcodec-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    espeak-ng

# Install all extras
pip install -e ".[dev]"
```

## Test Markers

| Marker | Meaning | When to use |
|--------|---------|-------------|
| (none) | Fast, no deps | Always runs in CI |
| `@pytest.mark.slow` | Downloads ML models | Skip with `-m "not slow"` |
| `@pytest.mark.gpu` | Requires CUDA | Skip with `-m "not gpu"` |

## Coverage by Category

| Category | Operators | Tests | Extras needed |
|----------|-----------|-------|---------------|
| basic | 4 | 4 | core |
| segment | 4 | 4 | `segment` |
| augment | 4 | 4 | `audio` |
| annotate | 18 | 18 | `asr`, `whisper`, `funasr`, `wenet`, `diarize`, `classify`, `speaker`, `enhance`, `align`, `codec` |
| quality | 11 | 11 | `pitch`, `dnsmos`, `quality` |
| pack | 6 | 6 | `pack` |
| synthesize | 4 | 4 | `tts-kokoro`, `tts-chattts`, `tts-cosyvoice`, `tts-fish-speech` |
| noop | 1 | 1 | core |

**Total: 51 operators, 51 test files, 100% coverage.**

## Test Patterns

Each operator test file follows the same structure:

1. **Import guard** — skip entire file if optional dep is missing
2. **Fast tests** — registration check + class attributes (no model needed)
3. **Slow tests** — `@pytest.mark.slow`, loads real model, runs `process()`

```python
# Example: tests/unit/operators/annotate/test_faster_whisper_asr.py

# 1. Import guard
pytest.importorskip("faster_whisper")

# 2. Fast
def test_faster_whisper_asr_is_registered(): ...
def test_faster_whisper_asr_class_attrs(): ...

# 3. Slow
@pytest.mark.slow
def test_faster_whisper_asr_transcribes(): ...
```

## What CI Runs

CI (`.github/workflows/ci.yml`) runs only fast tests:

```
pytest -v -m "not slow and not gpu" --cov=voxkitchen
```

Extras installed in CI: `segment`, `quality`, `pack`, `viz` (plus torch CPU).
Operators requiring other extras are skipped at module level.
