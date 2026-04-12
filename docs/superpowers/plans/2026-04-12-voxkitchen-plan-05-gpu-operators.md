# VoxKitchen Plan 5: Auto-Labeling Operators (GPU-capable, CPU-tested)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the final 5 operators: `silero_vad` (GPU-accelerated VAD), `faster_whisper_asr` (ASR transcription), `whisperx_asr` (word-aligned ASR), `pyannote_diarize` (speaker diarization), and `speechbrain_langid` + `speechbrain_gender` (audio classification). After this plan, all 22 v0.1 operators are complete.

**Development constraint: current dev machine has no GPU; production servers will have GPU.** Design for GPU as the primary target. For testing:
1. `device="gpu"` is correct — operators detect CUDA in `setup()` and fall back to CPU where the library supports it (silero, faster-whisper, speechbrain all work on CPU)
2. Tests that CAN run on CPU: mark `@pytest.mark.slow` (model downloads take time) — these prove the operator logic works
3. Tests that genuinely require GPU: mark `@pytest.mark.gpu` and skip honestly — no mocking to fake a pass
4. `pyannote_diarize` needs HuggingFace auth token — mark `@pytest.mark.gpu` (can only be properly tested on a configured server)

**Architecture:** Each operator follows the same pattern: `setup()` loads a model (detecting CPU/GPU), `process()` runs inference per Cut and adds Supervisions or metrics. Unlike Plan 3/4 operators, these are **annotation operators** — they don't produce new audio files (`produces_audio=False`). They read audio bytes, run a model, and add text/speaker/language/gender annotations to existing Cuts via new Supervisions or updated metrics.

**Tech Stack:** silero-vad (torch.hub), faster-whisper (CTranslate2), whisperx (faster-whisper + alignment), pyannote.audio (speaker diarization), speechbrain (language/gender classification).

---

## Key Design: CPU Fallback Pattern

Every GPU operator uses this pattern in `setup()`:

```python
def setup(self) -> None:
    import torch
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model to self._device
```

The class attribute `device = "gpu"` tells the runner to use `GpuPoolExecutor`, which spawns subprocesses with `CUDA_VISIBLE_DEVICES` set. On a CPU-only machine with `num_gpus: 1` (default), this spawns one subprocess where the operator detects no CUDA and uses CPU. This works transparently — no special handling needed.

## Key Design: Annotation Pattern

Unlike basic/segment/quality operators, auto-labeling operators **add Supervisions** to Cuts:

```python
def process(self, cuts: CutSet) -> CutSet:
    out = []
    for cut in cuts:
        audio, sr = load_audio_for_cut(cut)
        # Run model → get annotations (text, speaker, language, etc.)
        new_supervisions = self._run_model(audio, sr, cut)
        updated = cut.model_copy(update={
            "supervisions": [*cut.supervisions, *new_supervisions]
        })
        out.append(updated)
    return CutSet(out)
```

The Cut itself is not recreated with new provenance — annotations are appended to the existing Cut's supervisions list. This preserves the Cut's identity and provenance chain from upstream stages.

## Test Strategy

| Operator | Model size | CPU works? | Test approach |
|---|---|---|---|
| `silero_vad` | ~2 MB | Yes | Real model on CPU, `@pytest.mark.slow` |
| `faster_whisper_asr` | ~75 MB (tiny) | Yes (int8) | Real model on CPU, `@pytest.mark.slow` |
| `whisperx_asr` | ~75 MB + alignment | Yes (fallback) | Real model on CPU, `@pytest.mark.slow` |
| `pyannote_diarize` | ~100 MB + HF auth | Needs auth token | `@pytest.mark.gpu` — **skip on dev machine**, test on server |
| `speechbrain_langid` | ~30 MB | Yes | Real model on CPU, `@pytest.mark.slow` |
| `speechbrain_gender` | ~30 MB | Yes | Real model on CPU, `@pytest.mark.slow` |

Register markers in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that download ML models (deselect with '-m \"not slow\"')",
    "gpu: marks tests that require GPU or special server config (deselect with '-m \"not gpu\"')",
]
```

---

## File Structure

```
src/voxkitchen/operators/
├── segment/
│   └── silero_vad.py                # NEW: GPU-accelerated VAD
├── annotate/
│   ├── __init__.py
│   ├── faster_whisper_asr.py        # NEW: ASR transcription
│   ├── whisperx_asr.py              # NEW: word-aligned ASR
│   ├── pyannote_diarize.py          # NEW: speaker diarization
│   ├── speechbrain_langid.py        # NEW: language identification
│   └── speechbrain_gender.py        # NEW: gender classification

tests/unit/operators/
├── segment/
│   └── test_silero_vad.py           # NEW
├── annotate/
│   ├── __init__.py
│   ├── test_faster_whisper_asr.py   # NEW
│   ├── test_whisperx_asr.py         # NEW
│   ├── test_pyannote_diarize.py     # NEW
│   ├── test_speechbrain_langid.py   # NEW
│   └── test_speechbrain_gender.py   # NEW
```

---

## Task 1: Add auto-labeling dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Add new extras groups to `pyproject.toml`**

```toml
[project.optional-dependencies]
audio = ["torch>=2.1", "torchaudio>=2.1"]
segment = ["webrtcvad>=2.0", "librosa>=0.10"]
quality = ["simhash>=2.1"]
pack = ["datasets>=2.16", "webdataset>=0.2", "pyarrow>=14"]
asr = ["faster-whisper>=1.0"]
diarize = ["pyannote.audio>=3.1"]
classify = ["speechbrain>=1.0"]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "ruff>=0.4",
    "mypy>=1.8",
    "pre-commit>=3.6",
    "scipy>=1.11",
]
```

Notes:
- `silero-vad` is loaded via `torch.hub` — no separate pip package needed, just `torch` from the `audio` extras.
- `whisperx` wraps `faster-whisper` — add it to `asr` extras if it's a separate package, or just import it directly. Check if `whisperx` is pip-installable; if not, we implement the alignment ourselves using just faster-whisper output.
- `speechbrain` is one package covering both langid and gender.

- [ ] **Step 2: Add `slow` marker to pytest config**

In `pyproject.toml` `[tool.pytest.ini_options]`:
```toml
markers = ["slow: marks tests that download ML models (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 3: Update mypy overrides**

Add `faster_whisper.*`, `pyannote.*`, `speechbrain.*`, `whisperx.*` to the ignore_missing_imports module list.

- [ ] **Step 4: Update CI**

```yaml
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev,segment,quality,pack,asr,classify]"

      - name: Pytest (skip slow model downloads and GPU tests in CI)
        run: pytest -v -m "not slow and not gpu" --cov=voxkitchen --cov-report=term-missing
```

Note: CI skips `@pytest.mark.slow` tests to avoid downloading large models. Developers run `pytest` locally (without `-m`) to include model tests.

Also note: `diarize` extras (`pyannote.audio`) is NOT installed in CI because it requires HF auth. Pyannote tests are mocked anyway.

- [ ] **Step 5: Install locally**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,segment,quality,pack,asr,classify]"
```

Skip `diarize` unless you have a HuggingFace token set up.

- [ ] **Step 6: Verify + commit**

```bash
pytest -q -m "not slow"
```

All 189 tests pass. Commit: `chore: add asr/diarize/classify dependency extras + slow test marker`

---

## Task 2: `silero_vad` operator (TDD)

**Files:**
- Create: `src/voxkitchen/operators/segment/silero_vad.py`
- Create: `tests/unit/operators/segment/test_silero_vad.py`
- Modify: `src/voxkitchen/operators/__init__.py`

The spec's most important segmentation operator. Uses the Silero VAD model via `torch.hub`.

### Config

```python
class SileroVadConfig(OperatorConfig):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
```

### setup()

```python
def setup(self) -> None:
    import torch
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._model, self._utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    self._model.to(self._device)
```

### process()

For each Cut:
1. `load_audio_for_cut(cut)` → audio, sr
2. If sr != 16000: resample (silero requires 16kHz)
3. Call silero's `get_speech_timestamps(audio_tensor, model, ...)` utility
4. Each timestamp pair → a child Cut (same pattern as `fixed_segment`)

**Class attrs:** `name="silero_vad"`, `device="gpu"`, `produces_audio=False`, `reads_audio_bytes=True`

### Tests (4)

```python
@pytest.mark.slow
def test_silero_vad_detects_speech(mono_wav_16k: Path, tmp_path: Path) -> None:
    """Real model: a sine wave should produce at least 1 speech segment."""
    ...

@pytest.mark.slow
def test_silero_vad_child_cuts_have_provenance(mono_wav_16k: Path, tmp_path: Path) -> None:
    ...

def test_silero_vad_is_registered() -> None:
    assert get_operator("silero_vad") is SileroVadOperator

def test_silero_vad_class_attrs() -> None:
    assert SileroVadOperator.device == "gpu"
    assert SileroVadOperator.produces_audio is False
```

The first two tests download the model (~2MB, fast). Mark them `@pytest.mark.slow` anyway for consistency.

### Commit: `feat(operators): add silero_vad GPU-capable segmentation`

---

## Task 3: `faster_whisper_asr` operator (TDD)

**Files:**
- Create: `src/voxkitchen/operators/annotate/__init__.py`
- Create: `tests/unit/operators/annotate/__init__.py`
- Create: `src/voxkitchen/operators/annotate/faster_whisper_asr.py`
- Create: `tests/unit/operators/annotate/test_faster_whisper_asr.py`
- Modify: `src/voxkitchen/operators/__init__.py`

```bash
mkdir -p src/voxkitchen/operators/annotate
mkdir -p tests/unit/operators/annotate
```

### Config

```python
class FasterWhisperAsrConfig(OperatorConfig):
    model: str = "tiny"                # model size: tiny, base, small, medium, large-v3
    language: str | None = None        # None = auto-detect
    beam_size: int = 5
    compute_type: str = "int8"         # int8 for CPU, float16 for GPU
```

### setup()

```python
def setup(self) -> None:
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = self.config.compute_type
    if device == "cpu" and compute_type == "float16":
        compute_type = "int8"  # float16 not supported on CPU
    self._model = WhisperModel(
        self.config.model,
        device=device,
        compute_type=compute_type,
    )
```

### process()

For each Cut:
1. `load_audio_for_cut(cut)` → audio, sr
2. `segments, info = self._model.transcribe(audio, beam_size=..., language=...)`
3. For each segment: create a `Supervision` with `text`, `start`, `duration`, `language=info.language`
4. Append supervisions to the Cut

```python
new_sups = []
for seg in segments:
    new_sups.append(Supervision(
        id=f"{cut.id}__asr_{len(new_sups)}",
        recording_id=cut.recording_id,
        start=cut.start + seg.start,
        duration=seg.end - seg.start,
        text=seg.text.strip(),
        language=info.language,
    ))
updated = cut.model_copy(update={"supervisions": [*cut.supervisions, *new_sups]})
```

**Class attrs:** `name="faster_whisper_asr"`, `device="gpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["asr"]`

### Tests (4)

```python
def test_faster_whisper_asr_is_registered() -> None: ...
def test_faster_whisper_asr_class_attrs() -> None: ...

@pytest.mark.slow
def test_faster_whisper_asr_transcribes(mono_wav_16k, tmp_path) -> None:
    """Real tiny model on CPU: transcribe a sine wave. Text may be garbage but supervisions should exist."""
    ...
    result = op.process(cs)
    out_cut = next(iter(result))
    # Should have at least some supervisions
    assert len(out_cut.supervisions) > 0

@pytest.mark.slow
def test_faster_whisper_asr_adds_language(mono_wav_16k, tmp_path) -> None:
    """Supervisions from ASR should have a language field set."""
    ...
```

### Commit: `feat(operators): add faster_whisper_asr for speech transcription`

---

## Task 4: `whisperx_asr` operator (TDD)

**Files:**
- Create: `src/voxkitchen/operators/annotate/whisperx_asr.py`
- Create: `tests/unit/operators/annotate/test_whisperx_asr.py`
- Modify: `src/voxkitchen/operators/__init__.py`

WhisperX adds word-level timestamps on top of faster-whisper. If the `whisperx` package is not available, this operator falls back to `faster_whisper_asr` behavior (segment-level only).

### Config

```python
class WhisperxAsrConfig(OperatorConfig):
    model: str = "tiny"
    language: str | None = None
    batch_size: int = 8
    compute_type: str = "int8"
```

### setup()

```python
def setup(self) -> None:
    import torch
    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import whisperx
        self._whisperx = whisperx
        self._model = whisperx.load_model(
            self.config.model, self._device,
            compute_type=self.config.compute_type,
        )
        self._has_whisperx = True
    except ImportError:
        # Fallback: use faster-whisper directly
        from faster_whisper import WhisperModel
        self._model = WhisperModel(self.config.model, device=self._device, compute_type=self.config.compute_type)
        self._has_whisperx = False
```

### process()

If whisperx available: use `whisperx.transcribe()` → `whisperx.align()` for word-level alignment.
If fallback: same as `faster_whisper_asr` (segment-level).

**Class attrs:** `name="whisperx_asr"`, `device="gpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["asr"]`

### Tests (3)

```python
def test_whisperx_asr_is_registered() -> None: ...

@pytest.mark.slow
def test_whisperx_asr_transcribes(mono_wav_16k, tmp_path) -> None:
    """Falls back to faster-whisper if whisperx not installed."""
    ...
    assert len(out_cut.supervisions) > 0

@pytest.mark.slow
def test_whisperx_asr_supervisions_have_text(mono_wav_16k, tmp_path) -> None:
    ...
```

### Commit: `feat(operators): add whisperx_asr with word-level alignment`

---

## Task 5: `pyannote_diarize` operator (TDD — mocked model)

**Files:**
- Create: `src/voxkitchen/operators/annotate/pyannote_diarize.py`
- Create: `tests/unit/operators/annotate/test_pyannote_diarize.py`
- Modify: `src/voxkitchen/operators/__init__.py`

Speaker diarization: assigns speaker labels to time intervals.

**Problem:** pyannote.audio requires accepting a user agreement on HuggingFace and setting `HF_TOKEN`. This makes real-model tests impractical for automated testing. **Solution:** tests mock the pyannote pipeline.

### Config

```python
class PyannoteDiarizeConfig(OperatorConfig):
    model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int | None = None
    max_speakers: int | None = None
    hf_token: str | None = None      # or read from HF_TOKEN env var
```

### setup()

```python
def setup(self) -> None:
    import torch
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from pyannote.audio import Pipeline
    import os

    token = self.config.hf_token or os.environ.get("HF_TOKEN")
    self._pipeline = Pipeline.from_pretrained(
        self.config.model, use_auth_token=token
    )
    self._pipeline.to(self._device)
```

### process()

For each Cut:
1. Load audio → save to a temp wav (pyannote needs a file path or waveform dict)
2. Run pipeline: `diarization = self._pipeline({"waveform": tensor, "sample_rate": sr})`
3. For each turn in diarization: create a Supervision with `speaker` label

```python
for turn, _, speaker in diarization.itertracks(yield_label=True):
    new_sups.append(Supervision(
        id=f"{cut.id}__diar_{len(new_sups)}",
        recording_id=cut.recording_id,
        start=cut.start + turn.start,
        duration=turn.end - turn.start,
        speaker=speaker,
    ))
```

**Class attrs:** `name="pyannote_diarize"`, `device="gpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["diarize"]`

### Tests (3) — registration + attrs always run; functional test skipped without auth

```python
def test_pyannote_diarize_is_registered() -> None: ...

def test_pyannote_diarize_class_attrs() -> None:
    assert PyannoteDiarizeOperator.device == "gpu"
    assert PyannoteDiarizeOperator.produces_audio is False

@pytest.mark.gpu
def test_pyannote_diarize_adds_speaker_supervisions(mono_wav_16k, tmp_path) -> None:
    """Requires HF_TOKEN env var + pyannote model access. Run on configured server only."""
    ...
```

The functional test is marked `@pytest.mark.gpu` and honestly skipped on the dev machine. It will be tested on a GPU server with `HF_TOKEN` configured before release.

### Commit: `feat(operators): add pyannote_diarize for speaker diarization`

---

## Task 6: `speechbrain_langid` + `speechbrain_gender` (TDD)

**Files:**
- Create: `src/voxkitchen/operators/annotate/speechbrain_langid.py`
- Create: `src/voxkitchen/operators/annotate/speechbrain_gender.py`
- Create: `tests/unit/operators/annotate/test_speechbrain_langid.py`
- Create: `tests/unit/operators/annotate/test_speechbrain_gender.py`
- Modify: `src/voxkitchen/operators/__init__.py`

Both use SpeechBrain's pretrained classifiers.

### `speechbrain_langid`

**Config:**
```python
class SpeechBrainLangIdConfig(OperatorConfig):
    model: str = "speechbrain/lang-id-voxlingua107-ecapa"
```

**setup():** Load the classifier:
```python
from speechbrain.inference.classifiers import EncoderClassifier
self._classifier = EncoderClassifier.from_hparams(
    source=self.config.model, run_opts={"device": str(self._device)}
)
```

**process():** For each Cut:
1. Load audio, ensure 16kHz mono
2. `prediction = self._classifier.classify_batch(tensor)`
3. Extract language label
4. Update `cut.supervisions` with language, or directly set language on existing supervisions

Simpler approach: just add the language to all supervisions in the cut, or add it to `cut.custom["language"]`:
```python
updated = cut.model_copy(update={"custom": {**cut.custom, "language": lang}})
```

Actually, per spec, the Supervision model has a `language` field. So add a new Supervision:
```python
Supervision(id=f"{cut.id}__langid", recording_id=cut.recording_id, start=cut.start, duration=cut.duration, language=lang)
```

**Class attrs:** `name="speechbrain_langid"`, `device="gpu"`, `produces_audio=False`, `reads_audio_bytes=True`, `required_extras=["classify"]`

### `speechbrain_gender`

Same pattern but classifies gender (male/female).

**Config:**
```python
class SpeechBrainGenderConfig(OperatorConfig):
    model: str = "speechbrain/spkrec-xvect-voxceleb"  # or a gender-specific model
```

**process():** Similar to langid but sets `gender` on Supervision.

**Note:** SpeechBrain doesn't have a dedicated "gender classification" model out of the box. The common approach is to use a speaker embedding model and train a simple classifier on top. For v0.1, we can:
- Use a speaker verification model to extract embeddings, then apply a simple heuristic
- OR just use a pretrained model if one exists
- OR simplify: mark the classification as "experimental" and use whatever SpeechBrain provides

For simplicity, let's make `speechbrain_gender` use the same classifier API but with a gender-specific model path. If no reliable model exists, the operator can produce a placeholder classification (e.g., always "unknown") with a warning, and the real model can be swapped in later.

### Tests (4 per operator, 8 total)

```python
def test_speechbrain_langid_is_registered() -> None: ...
def test_speechbrain_langid_class_attrs() -> None: ...

@pytest.mark.slow
def test_speechbrain_langid_classifies(mono_wav_16k, tmp_path) -> None:
    """Real model: classify language of a 1s sine wave. Result may be arbitrary but should be a valid language code."""
    ...

@pytest.mark.slow
def test_speechbrain_langid_adds_supervision_with_language(mono_wav_16k, tmp_path) -> None:
    ...
```

Same pattern for gender (4 tests). If the gender model is unreliable, mark the test as `@pytest.mark.slow` and just verify the operator runs without error.

### Two commits:
- `feat(operators): add speechbrain_langid for language identification`
- `feat(operators): add speechbrain_gender for gender classification`

---

## Task 7: Integration test — ASR pipeline

**Files:**
- Create: `tests/integration/test_asr_pipeline_e2e.py`

A pipeline exercising the annotation operators: `dir_scan → silero_vad → faster_whisper_asr → pack_manifest`.

```python
@pytest.mark.slow
def test_vad_asr_pipeline(audio_dir: Path, tmp_path: Path) -> None:
    """Scan → Silero VAD → Faster Whisper ASR → Pack. All on CPU."""
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(f"""
version: "0.1"
name: asr-e2e
work_dir: {work_dir}
num_gpus: 1
num_cpu_workers: 1
ingest:
  source: dir
  args:
    root: {audio_dir}
    recursive: true
stages:
  - name: vad
    op: silero_vad
    args:
      threshold: 0.3
  - name: asr
    op: faster_whisper_asr
    args:
      model: tiny
      compute_type: int8
  - name: pack
    op: pack_manifest
""")
    spec = load_pipeline_spec(yaml_path, run_id="run-asr-e2e")
    run_pipeline(spec)

    # All stages complete
    assert (work_dir / "00_vad" / "_SUCCESS").exists()
    assert (work_dir / "01_asr" / "_SUCCESS").exists()
    assert (work_dir / "02_pack" / "_SUCCESS").exists()

    # ASR should have added supervisions with text
    final_cuts = list(read_cuts(work_dir / "02_pack" / "cuts.jsonl.gz"))
    assert len(final_cuts) > 0
    # At least some cuts should have supervisions with text
    cuts_with_text = [c for c in final_cuts if any(s.text for s in c.supervisions)]
    # Note: a sine wave may not produce meaningful ASR output, but the pipeline should complete
```

This test is `@pytest.mark.slow` (downloads both silero and whisper-tiny models).

### Commit: `test(integration): add VAD → ASR pipeline test (CPU, slow)`

---

## Task 8: Full verification, lint, type, tag

- [ ] `pytest -m "not slow and not gpu" -v` — all fast tests pass (registration + class attrs)
- [ ] `pytest -m "not gpu" -v` (optional, slow) — model-download tests pass on CPU
- [ ] `pytest -v` (on GPU server only) — full suite including pyannote
- [ ] `ruff check src tests` + `ruff format --check src tests`
- [ ] `mypy src/voxkitchen tests`
- [ ] `pre-commit run --all-files`
- [ ] Tag: `git tag -a plan-05-gpu-operators -m "Plan 5 complete: all 22 operators (6 auto-labeling, CPU-tested)"`

---

## Plan 5 Completion Checklist

- [ ] All 6 new operators registered: `silero_vad`, `faster_whisper_asr`, `whisperx_asr`, `pyannote_diarize`, `speechbrain_langid`, `speechbrain_gender`
- [ ] All operators fall back to CPU when CUDA unavailable (verified in tests)
- [ ] `silero_vad` creates child Cuts (same pattern as `fixed_segment`/`webrtc_vad`)
- [ ] `faster_whisper_asr` adds Supervisions with `text` and `language`
- [ ] `whisperx_asr` falls back to faster-whisper if whisperx not installed
- [ ] `pyannote_diarize` adds Supervisions with `speaker` labels (mocked in tests)
- [ ] `speechbrain_langid` adds language classification
- [ ] `speechbrain_gender` adds gender classification
- [ ] `@pytest.mark.slow` on CPU-capable model-download tests
- [ ] `@pytest.mark.gpu` on tests requiring GPU or HF auth (pyannote)
- [ ] CI runs with `-m "not slow and not gpu"` (no model downloads, no GPU)
- [ ] `pytest -m "not gpu"` passes locally (including slow CPU model tests)
- [ ] All existing 189 non-model tests still pass
- [ ] `git tag plan-05-gpu-operators` at HEAD

### Operator tally after Plan 5: 22/22 complete

| Category | Count | All done? |
|---|---|---|
| Basic | 4 | Yes |
| Segment | 4 | Yes (+silero_vad) |
| Annotate | 5 | Yes |
| Quality | 4 | Yes |
| Pack | 5 | Yes |

## What Plans 6-8 Will Build On

**Plan 6 (Ingest recipes):** LibriSpeech, CommonVoice, AISHELL-1. Download + convert.
**Plan 7 (Visualization):** Rich inspect, HTML report, Gradio panel.
**Plan 8 (Plugin + polish):** entry_points, `vkit init`, docs, release.

No more operators after Plan 5. The remaining plans are about user-facing features and polish.
