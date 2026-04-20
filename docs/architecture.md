# System Architecture

> Operators: 51 across 7 categories.

## Overview

VoxKitchen is a declarative speech data processing toolkit. Users write a YAML pipeline, run `vkit run`, and get training-ready datasets with full provenance tracking.

**Core metaphor:** pipeline.yaml is a recipe, operators are cooking steps, ingest recipes are ingredient prep, `pack` is plating.

**Target users:** Speech researchers (ASR, TTS, speaker recognition, speech LLMs).

**Key guarantees:** Reproducible, resumable, inspectable pipelines. Every stage checkpoints to disk; crashes resume from the last completed stage.

---

## Layered Architecture

```
CLI Layer           (cli/)           User-facing commands
    |
Pipeline Layer      (pipeline/)      Orchestration, execution, GC
    |
Operator Layer      (operators/)     51 built-in transformations
    |
Schema Layer        (schema/)        Pydantic v2 data models
```

**Dependency rule:** Each layer depends only on layers below it. `operators/` imports from `schema/` only; `RunContext` is imported under `TYPE_CHECKING` to avoid circular deps.

---

## Data Model (Schema Layer)

All types are Pydantic v2 `BaseModel` subclasses with `extra="forbid"`.

### Recording

Physical audio resource. Immutable after creation.

```
Recording
  id: str
  sources: list[AudioSource]       # type: file|url|command
  sampling_rate: int
  num_samples: int
  duration: float
  num_channels: int
  checksum: str | None
  custom: dict[str, Any]
```

### Supervision

Time-aligned annotation over a Recording. Fields are progressively filled: VAD creates supervisions without `text`; ASR adds `text` later.

```
Supervision
  id, recording_id, start, duration    # required
  text, language, speaker, gender      # optional (filled by operators)
  channel, age_range, custom           # optional
```

### Cut

The unit flowing through pipelines. References a `[start, start+duration)` slice of a Recording plus overlapping Supervisions. **Immutable** -- operators produce new Cuts, never mutate existing ones.

```
Cut
  id, recording_id, start, duration    # identity
  recording: Recording | None         # embedded for audio operators
  supervisions: list[Supervision]
  metrics: dict[str, float]            # snr, cer, utmos, ...
  custom: dict[str, Any]               # embeddings, tokens, ...
  provenance: Provenance               # lineage tracking
```

### CutSet

Collection of Cuts. Serialized as gzip-compressed JSONL (`.jsonl.gz`). First line is a header with schema version.

Operations: `filter()`, `map()`, `split(n)`, `merge()`, lazy iteration.

### Provenance

Every Cut records how it was produced:

```
Provenance
  source_cut_id: str | None       # parent Cut (None for ingested)
  generated_by: str                # e.g. "silero_vad"
  stage_name: str                  # e.g. "02_vad"
  created_at: datetime
  pipeline_run_id: str
```

Enables `vkit inspect trace <cut-id>` to reconstruct the full lineage chain.

---

## Operator System

### Base Contract

```python
class Operator(ABC):
    name: ClassVar[str]                         # unique registry key
    config_cls: ClassVar[type[OperatorConfig]]   # Pydantic config model
    device: ClassVar[Literal["cpu", "gpu"]]      # execution target
    produces_audio: ClassVar[bool]               # creates new WAV files?
    reads_audio_bytes: ClassVar[bool]            # needs raw audio samples?
    required_extras: ClassVar[list[str]]         # pip extras needed

    def setup(self) -> None: ...       # load models (once per worker)
    def process(self, cuts: CutSet) -> CutSet: ...  # transform
    def teardown(self) -> None: ...    # release resources
```

### Registration

Built-in operators: `@register_operator` decorator + import in `operators/__init__.py`.
Third-party: `entry_points` group `voxkitchen.operators` (lazy discovery on first access).

Optional deps wrapped in `try/except ImportError` -- missing packages don't crash the core.

### Operator Catalog (51 operators, 7 categories)

| Category | Count | Operators |
|----------|-------|-----------|
| **Audio** | 5 | `resample`, `ffmpeg_convert`, `channel_merge`, `loudness_normalize`, `identity` |
| **Segmentation** | 4 | `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` |
| **Augmentation** | 4 | `speed_perturb`, `volume_perturb`, `noise_augment`, `reverb_augment` |
| **Synthesize** | 4 | `tts_kokoro`, `tts_chattts`, `tts_cosyvoice`, `tts_fish_speech` |
| **Annotation** | 17 | `faster_whisper_asr`, `whisper_openai_asr`, `whisperx_asr`, `paraformer_asr`, `sensevoice_asr`, `wenet_asr`, `qwen3_asr`, `pyannote_diarize`, `speechbrain_langid`, `whisper_langid`, `gender_classify`, `speaker_embed`, `speech_enhance`, `forced_align`, `emotion_recognize`, `codec_tokenize`, `mel_extract` |
| **Quality** | 11 | `snr_estimate`, `dnsmos_score`, `utmos_score`, `pitch_stats`, `clipping_detect`, `bandwidth_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter`, `speaker_similarity`, `cer_wer` |
| **Pack** | 6 | `pack_manifest`, `pack_jsonl`, `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` |

### Operator Patterns

**Analysis operator** (e.g., `snr_estimate`): reads audio, writes to `metrics`. `produces_audio=False, reads_audio_bytes=True`.

**Audio-producing operator** (e.g., `resample`, `speech_enhance`): reads audio, writes new WAV to `derived/`, creates new Recording + Cut. `produces_audio=True, reads_audio_bytes=True`.

**Text-only operator** (e.g., `cer_wer`): reads from `supervision.text` or `custom`. No audio access. `reads_audio_bytes=False`.

**TTS synthesis operator** (e.g., `tts_kokoro`): reads text from `supervision.text`, generates audio in `derived/`. `produces_audio=True, reads_audio_bytes=False`.

---

## Pipeline Engine

### YAML Spec

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${name}-${run_id}     # variable interpolation
num_gpus: 1
num_cpu_workers: null                   # auto-detect
gc_mode: aggressive                     # aggressive | keep

ingest:
  source: dir | manifest | recipe
  args: { root: ./data }

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000 }
  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }
```

Supports `${name}`, `${run_id}`, `${env:VAR}` interpolation in all string values.

### Execution Flow

```
vkit run pipeline.yaml
  |
  v
[Loader] YAML -> PipelineSpec (Pydantic validation + interpolation)
  |
  v
[Runner] Resume check -> find last completed stage
  |
  v
[Ingest] DirScan | Manifest | Recipe -> initial CutSet
  |
  v
[Stage Loop]
  For each stage:
    1. Instantiate operator + config
    2. Select executor (CPU pool or GPU pool)
    3. Shard CutSet across workers
    4. Workers: setup() -> process() -> teardown()
    5. Write cuts.jsonl.gz + _SUCCESS marker
    6. Run GC on expired derived audio
  |
  v
[Finalize] Generate report, empty trash
```

### Work Directory Layout

```
work_dir/
  run.yaml                    # spec snapshot
  00_resample/
    cuts.jsonl.gz             # output manifest
    _SUCCESS                  # completion marker
    _errors.jsonl             # per-cut errors (if any)
    _stats.json               # timing, throughput
    derived/                  # new audio files (if produces_audio)
  01_vad/
    cuts.jsonl.gz
    _SUCCESS
    ...
  derived_trash/              # GC'd audio (emptied on success)
```

### Executors

**CpuPoolExecutor:** `multiprocessing.Pool` with spawn context. Shards CutSet, runs operator per shard. Config passed as JSON (not pickled) for cross-process safety.

**GpuPoolExecutor:** Spawns N subprocesses, each pinned to one GPU via `CUDA_VISIBLE_DEVICES=i` before torch import. Operator sees `cuda:0`.

**Error handling:** If a shard fails, retries cut-by-cut. Bad cuts logged to `_errors.jsonl`, pipeline continues.

### Resume & Checkpointing

A stage is complete iff both `cuts.jsonl.gz` and `_SUCCESS` exist. `_SUCCESS` is written atomically after the manifest is fully flushed.

```bash
vkit run pipeline.yaml                    # full run
vkit run pipeline.yaml --resume-from vad  # resume from stage
```

### Garbage Collection

Static analysis builds a GC plan: for each `produces_audio=True` stage, find its last downstream consumer (`reads_audio_bytes=True`). After that consumer completes, move the producer's `derived/` to trash. Trash emptied only on successful pipeline completion.

`gc_mode: keep` or `--keep-intermediates` disables GC.

---

## Ingest Sources

| Source | Input | Use Case |
|--------|-------|----------|
| `dir` | Directory of audio files | Raw audio processing |
| `manifest` | Existing `cuts.jsonl.gz` | Resume / chain pipelines |
| `recipe` | Named dataset parser | Standard datasets |

### Built-in Recipes

- `librispeech` -- LibriSpeech ASR corpus
- `aishell` -- AISHELL-1 Mandarin speech
- `commonvoice` -- Mozilla Common Voice
- `fleurs` -- Google FLEURS multilingual

Each recipe implements `download()` and `prepare(root, subsets, ctx) -> CutSet`.

---

## CLI Commands

| Command | Purpose |
|---------|---------|
| `vkit init <path> [-t template]` | Scaffold project |
| `vkit run <yaml>` | Execute pipeline |
| `vkit validate <yaml>` | Validate without executing |
| `vkit download <recipe>` | Download dataset |
| `vkit ingest` | Standalone ingest |
| `vkit inspect run <dir>` | Stage summary |
| `vkit inspect cuts <path>` | Cut statistics |
| `vkit inspect trace <id>` | Provenance chain |
| `vkit inspect errors <dir>` | Error report |
| `vkit operators` | List operators |
| `vkit operators show <name>` | Operator detail |
| `vkit recipes` | List recipes |
| `vkit viz <path>` | Gradio explorer |

Templates: `tts`, `asr`, `cleaning`, `speaker` (stored in `examples/pipelines/`).

---

## Python Tools API

For one-off tasks without writing YAML:

```python
from voxkitchen.tools import (
    transcribe, detect_speech, estimate_snr,
    extract_speaker_embedding, enhance_speech,
    align_words, synthesize,
)

transcribe("speech.wav", model="large-v3")
detect_speech("speech.wav", method="silero")
estimate_snr("speech.wav")
extract_speaker_embedding("speaker.wav")
enhance_speech("noisy.wav", "clean.wav")
align_words("speech.wav", "hello world")
synthesize("Hello!", "output.wav", engine="kokoro")
```

Each function creates a temporary Cut + RunContext, runs the corresponding operator, and returns the result.

---

## Dependency Management

**Core** (always installed): pydantic, pyyaml, typer, rich, soundfile, numpy, ffmpeg-python, pyloudnorm

**Optional extras** (install what you need):

| Group | Packages | For |
|-------|----------|-----|
| `audio` | torch, torchaudio | Resample, VAD |
| `asr` | faster-whisper | ASR transcription |
| `segment` | webrtcvad, librosa | Speech segmentation |
| `speaker` | wespeaker | Speaker embeddings |
| `diarize` | pyannote.audio | Speaker diarization |
| `enhance` | deepfilternet | Speech denoising |
| `align` | qwen-asr | Forced alignment |
| `codec` | encodec, dac | Neural codec tokens |
| `tts-kokoro` | kokoro, misaki | Kokoro TTS (CPU) |
| `tts-chattts` | ChatTTS | ChatTTS (GPU) |
| `tts-cosyvoice` | modelscope | CosyVoice2 (GPU) |
| `tts-fish-speech` | fish-speech | Fish-Speech (GPU) |
| `viz` | jinja2, plotly | HTML report |
| `viz-panel` | gradio | Interactive panel |

---

## Project Structure

```
voxkitchen/
  cli/                  # Typer CLI app
  operators/            # 51 operators across 7 categories
    basic/              #   resample, ffmpeg_convert, ...
    segment/            #   silero_vad, webrtc_vad, ...
    augment/            #   speed_perturb, noise_augment, ...
    synthesize/         #   tts_kokoro, tts_chattts, ...
    annotate/           #   faster_whisper_asr, speaker_embed, ...
    quality/            #   snr_estimate, cer_wer, ...
    pack/               #   pack_jsonl, pack_huggingface, ...
    noop/               #   identity
  pipeline/             # Runner, executors, checkpoint, GC
  schema/               # Cut, CutSet, Recording, Supervision, Provenance
  ingest/               # DirScan, Manifest, Recipe sources
    recipes/            #   librispeech, aishell, commonvoice, fleurs
  viz/                  # HTML report + Gradio panel
  templates/            # vkit init template registry
  plugins/              # Entry-point discovery
  utils/                # Audio I/O, time, download helpers
  tools.py              # Standalone tool functions

examples/pipelines/     # 20+ example YAML pipelines
tests/unit/             # 297+ unit tests
tests/integration/      # End-to-end pipeline tests
```

---

## Design Principles

1. **Immutability** -- Operators create new Cuts, never mutate existing ones
2. **Resumability** -- All state on disk; `_SUCCESS` markers enable crash recovery
3. **Error tolerance** -- Bad Cuts logged and skipped, pipeline continues
4. **Minimal core** -- Heavy deps (torch, transformers) are optional extras
5. **Provenance** -- Every Cut tracks its lineage
6. **Declarative first** -- YAML is the primary interface; Python API is the escape hatch
7. **Simplicity over efficiency** -- Resolve conflicts in favor of simplicity

---

## Roadmap

### Completed

- Core framework (schema, pipeline engine, CLI)
- 51 operators across 7 categories
- 4 ingest recipes (LibriSpeech, AISHELL, CommonVoice, FLEURS)
- Visualization (Rich CLI, HTML report, Gradio panel)
- Plugin system (entry_points)
- TTS synthesis (Kokoro, ChatTTS, CosyVoice2, Fish-Speech)

### Planned

- **Distributed execution** -- Ray/Dask backend for multi-node pipelines
- **Cloud storage** -- S3/GCS as audio source and output target
- **Dataset versioning** -- DVC integration for manifest version control
- **Additional recipes** -- GigaSpeech, WenetSpeech, MLS, VoxCeleb
- **Streaming pipelines** -- Process audio streams without full materialization
- **Training integration** -- Direct export to training frameworks (NeMo, ESPnet, WeNet)
