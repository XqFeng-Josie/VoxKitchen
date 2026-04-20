---
name: voxkitchen
description: >
  Expert assistant for VoxKitchen — a declarative speech data processing toolkit.
  Use this skill whenever the user wants to build audio/speech processing pipelines,
  create YAML recipes for TTS/ASR/speaker data preparation, choose from VoxKitchen's
  51 operators, run the `vkit` CLI, debug or inspect pipeline results, or use the
  Python tools API for one-off audio tasks. Trigger on: vkit, voxkitchen, speech pipeline,
  audio dataset prep, TTS data, ASR training data, VAD, speaker diarization pipeline,
  audio YAML recipe, or any mention of processing audio at scale. When in doubt, use
  this skill — it knows the operators, YAML schema, CLI flags, and common pitfalls.
---

# VoxKitchen Skill

VoxKitchen is a **declarative speech data processing toolkit**. Users write YAML pipeline
recipes, run `vkit run`, and get training-ready datasets with full provenance tracking.

**Key mental model:** A pipeline is a series of *stages*, each applying an *operator* to
a *CutSet* (collection of audio segments). Operators are composable building blocks — 51
are built in. Pipelines checkpoint every stage so they survive crashes and can resume.

**Project location:** `/data/xiaoqinfeng/code/VoxKitchen/`

---

## How to approach requests

### 1. Creating or designing a pipeline

When the user wants to build a pipeline, ask:
- What's the **goal**? (TTS data prep / ASR training data / data cleaning / speaker analysis)
- What's the **data source**? (local audio files / pre-built manifest / public dataset recipe)
- What **quality bar** is needed? (SNR threshold, duration range, CER limit, etc.)
- **Local execution or Docker?** (Does the pipeline cross dependency clusters — e.g., need both TTS synthesis and ASR? If so, recommend Docker.)

Then scaffold the pipeline using the YAML schema below. Start from the matching template
(`vkit init --template <tts|asr|cleaning|speaker>`) when possible, then customize stages.

**Warn users:** never `pip install .[all]` — dependency clusters conflict. Each cluster
(core / asr / diarize / tts / fish-speech) must be installed independently.

### 2. Choosing operators

Operators are grouped by function. When helping pick operators, think about the data flow:
`Ingest → Audio prep → Segmentation → Augmentation → Annotation → Quality filtering → Pack`

Read `references/operators.md` for the full catalog with config fields. Key decisions:
- **VAD:** `silero_vad` (default, robust) vs `webrtc_vad` (fast, less accurate)
- **ASR:** `faster_whisper_asr` (GPU, multilingual) / `qwen3_asr` (best Chinese) / `paraformer_asr` (streaming)
- **Quality filter:** `quality_score_filter` accepts Python-like condition strings on duration, metrics.snr, etc.
- **TTS:** `tts_kokoro` (English, high quality) / `tts_cosyvoice` (Chinese + voice clone) / `tts_fish_speech` (isolated cluster)
- **Pack output:** `pack_huggingface` (HF datasets), `pack_webdataset` (streaming), `pack_kaldi` (Kaldi), `pack_parquet`

### 3. Running and debugging pipelines

Workflow:
```bash
vkit validate pipeline.yaml          # Validate YAML + check operator args first
vkit run pipeline.yaml --dry-run     # Simulate without executing
vkit run pipeline.yaml               # Execute
vkit inspect run ./work/             # Check per-stage stats after run
```

**Resuming after crash:** each stage writes `_SUCCESS` marker. Re-running continues from
the last completed stage automatically. Force a restart from a specific stage:
```bash
vkit run pipeline.yaml --resume-from 03_asr
```

**Common issues and fixes:**
- `pyannote_diarize` returns 403 → HF_TOKEN missing or model agreement not accepted
- `silero_vad` hangs on first run → pre-download the model: `python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"`
- `faster_whisper_asr` deadlocks on macOS → use `whisper_openai_asr` instead
- Pipeline needs TTS + ASR operators together → use `vkit docker run pipeline.yaml --tag latest`

**Check environment health:**
```bash
vkit doctor                          # Shows which operators are available
vkit doctor --expect asr             # Asserts ASR operators importable
```

### 4. Inspecting pipeline results

```bash
vkit inspect run ./work/             # Per-stage: counts, duration hours, errors, timing
vkit inspect cuts work/05_pack/cuts.jsonl.gz   # CutSet statistics
vkit inspect errors ./work/          # What failed and why
vkit inspect trace <cut-id> --in ./work/       # Provenance chain for a specific cut
vkit viz cuts.jsonl.gz --port 7860   # Interactive Gradio visualization
```

Each stage directory contains:
- `cuts.jsonl.gz` — the CutSet at that stage
- `_stats.json` — counts, duration, timing
- `_errors.jsonl` — failed cuts (pipeline continues past errors)
- `_SUCCESS` — marker file (stage complete)

### 5. Python tools API (quick one-off tasks)

For tasks that don't need a full pipeline, the tools API is faster:

```python
from voxkitchen.tools import (
    audio_info, transcribe, detect_speech, estimate_snr,
    extract_speaker_embedding, enhance_speech, align_words, synthesize,
)

info = audio_info("speech.wav")                    # duration, sample_rate, format
segments = transcribe("speech.wav", model="tiny")  # → [SpeechSegment(start, end, text)]
speech = detect_speech("speech.wav", method="silero")
snr = estimate_snr("speech.wav")                   # float in dB
enhance_speech("noisy.wav", "clean.wav")
words = align_words("speech.wav", "hello world")   # forced alignment
synthesize("Hello!", "out.wav", engine="kokoro")
# Voice clone:
synthesize("你好", "out.wav", engine="cosyvoice",
           reference_audio="ref.wav", reference_text="参考文本")
```

---

## YAML Pipeline Quick Reference

```yaml
version: "0.1"
name: my-pipeline
description: "Optional description"
work_dir: ./work/${name}-${run_id}   # ${env:VAR_NAME} also works
num_gpus: 1
num_cpu_workers: null                # auto by default
gc_mode: aggressive                  # "aggressive" (clean intermediate audio) or "keep"

ingest:
  source: dir | manifest | recipe
  recipe: librispeech                # for source=recipe only
  args:
    root: /path/to/audio             # for dir/recipe
    path: /path/to/cuts.jsonl.gz     # for manifest
    recursive: true

stages:
  - name: stage_name                 # unique, used in work_dir subdirs
    op: operator_name                # from the operator catalog
    args:
      param: value
```

**Ingest sources:**
- `dir` — scan directory for audio files (`args.root`, `args.recursive`)
- `manifest` — load existing CutSet (`args.path`)
- `recipe` — download/load public dataset (`recipe: librispeech|aishell|commonvoice|fleurs`, `args.root`, `args.subsets`)

For detailed operator configs and all 51 operators, read `references/operators.md`.
For full YAML schema details, read `references/pipeline-yaml.md`.
For CLI flags reference, read `references/cli-reference.md`.

---

## Template Pipelines

Use `vkit init <path> --template <name>` to scaffold these:

| Template | Stages |
|---|---|
| `tts` | ffmpeg_convert → resample → denoise → vad → snr → filter → asr → pack |
| `asr` | resample → vad → speed_aug → volume_aug → asr → filter → pack |
| `cleaning` | quality metrics → dedup → filter → pack |
| `speaker` | vad → diarize → embeddings → gender → pack |

When helping a user build a custom pipeline, use the closest template as a starting point
and explain what each stage does and why it's placed in that order.

---

## Operator Categories at a Glance

| Category | Count | Key operators |
|---|---|---|
| Audio | 5 | resample, ffmpeg_convert, loudness_normalize |
| Segmentation | 4 | silero_vad, webrtc_vad, fixed_segment |
| Augmentation | 4 | speed_perturb, volume_perturb, noise_augment |
| Annotation | 17 | faster_whisper_asr, qwen3_asr, pyannote_diarize, speaker_embed |
| Quality | 11 | snr_estimate, dnsmos_score, quality_score_filter, audio_fingerprint_dedup |
| Synthesize | 4 | tts_kokoro, tts_cosyvoice, tts_fish_speech, tts_chattts |
| Pack | 6 | pack_huggingface, pack_webdataset, pack_parquet, pack_kaldi |

Read `references/operators.md` for config fields, required extras, and device requirements.

---

## Docker Execution

Use Docker when:
- The pipeline crosses dependency clusters (e.g., TTS + ASR together)
- Users want GPU operators without managing CUDA environments
- Reproducibility is critical

```bash
vkit docker pull --tag latest        # ~103 GB, all envs
vkit docker pull --tag slim          # ~13 GB, CPU only
vkit docker run pipeline.yaml        # runs in :latest
vkit docker run pipeline.yaml --tag asr --gpus none   # CPU-only ASR image
vkit docker shell --tag asr          # interactive bash in container
vkit docker build asr                # build local image
```

Docker auto-mounts `./work` and `./data`, sets user to host UID, and loads `.env` for HF_TOKEN.

---

## Environment Setup

```bash
# Core cluster (always needed)
pip install -e ".[audio,segment,quality,pack]"

# Add ASR support
pip install -e ".[audio,segment,quality,pack,asr,align]"

# Add diarization (separate cluster)
pip install -e ".[audio,segment,quality,pack,diarize]"

# Add TTS (separate cluster)
pip install -e ".[audio,segment,quality,pack,tts-kokoro,tts-cosyvoice]"
```

Set `HF_TOKEN` in `.env` for pyannote diarization. Copy `.env.example` as starting point.

---

## When generating YAML for users

- Always include `version: "0.1"`, `name`, `work_dir`, `ingest`, and at least one stage
- Use `${run_id}` in work_dir to avoid overwriting runs
- End pipelines with a `pack_*` stage so results are in a usable format
- Validate the YAML mentally: stage names must be unique, operator names must be exact
- Add a `quality_score_filter` stage before expensive operators (ASR, TTS) to skip bad audio early
- For `quality_score_filter`, conditions are Python-like strings: `"duration > 2"`, `"metrics.snr > 15"`, `"metrics.utmos > 3.5"`
- Always tell the user to run `vkit validate pipeline.yaml` before executing
