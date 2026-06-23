# Testing Guide

## Quick Reference

```bash
# Local CI gate before push
scripts/check-ci.sh

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

# Fast full suite (all modules, not just operators)
pytest -v -m "not slow and not gpu"
```

## Docker Runtime Smoke Checks

Runtime images are built from `docker/Dockerfile`. Use Docker smoke checks
to validate image health; run the pytest suite from the local dev
environment below. `vkit docker build` keeps Docker client temp/config/cache
files under `./.docker` by default; set `VKIT_DOCKER_WORK_DIR` to override it.

```bash
# Build the small core image and run its doctor check
vkit docker build slim
vkit docker doctor --tag slim --expect core

# Build the full multi-env image and inspect all envs
vkit docker build latest
vkit docker doctor --tag latest

# Debug inside an image
vkit docker shell --tag latest --gpus all
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

# Install local test tooling; real pipeline runs use Docker images.
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
| utility (`noop`) | 1 | 1 root-level operator test | core |

**Total: 52 operators.** Each built-in operator has at least one unit test;
shared base and registry tests are additional.

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

CI (`.github/workflows/ci.yml`) and `scripts/check-ci.sh` run the same fast
gate:

```
ruff check voxkitchen tests
ruff format --check voxkitchen tests
mypy voxkitchen
pytest -m "not slow and not gpu" --cov=voxkitchen \
  --deselect tests/unit/operators/pack/test_pack_huggingface.py \
  --deselect tests/unit/operators/pack/test_pack_parquet.py \
  --deselect tests/unit/viz/test_report.py
```

Extras installed in CI: `audio`, `segment`, `quality`, `pack`, `pitch`,
`dnsmos`, `classify`, `enhance`, and `viz` (plus torch CPU). Operators
requiring other extras are skipped at module level.
