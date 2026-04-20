# Getting Started

## Installation

VoxKitchen has one CLI (`vkit`) with two execution modes: **local** (runs
inside your current Python env) and **container** (runs inside a Docker
image). Most users use both — pip install to get the CLI, then pull
Docker images for pipelines that exceed the local env's extras.

### Local install (pip)

```bash
# Create virtual environment
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen

# Clone + install
git clone https://github.com/XqFeng-Josie/VoxKitchen.git
cd VoxKitchen
pip install -e ".[audio,segment,quality,pack]"   # pick one dep cluster

# Add more extras from the same cluster as needed
pip install -e ".[audio,segment,quality,pack,asr,funasr,align]"   # ASR cluster
pip install -e ".[audio,segment,quality,pack,diarize]"            # diarization
pip install -e ".[audio,segment,quality,pack,tts-kokoro]"         # TTS
```

> **Don't install `.[all]`** — it crosses dep clusters (pyannote vs
> funasr vs fish-speech pin incompatible torch/numpy versions) and
> the pip resolver will fail. The authoritative extras → cluster
> mapping is
> [`voxkitchen/runtime/env_resolver.py`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/voxkitchen/runtime/env_resolver.py).

### Container install (Docker)

For pipelines that cross dep clusters (e.g. pyannote + funasr), use
the published multi-env image:

```bash
vkit docker pull --tag slim          # CPU-only, ~13 GB
vkit docker pull --tag latest        # all five envs, ~103 GB
vkit docker run pipeline.yaml        # same YAML, same CLI
```

Full tag matrix and size reference: [Docker build guide](docker-build.md).

## Your first pipeline

### Option A: Use a template (recommended)

```bash
vkit init my-project --template tts    # TTS data preparation
cd my-project
```

Available templates:

| Template | Use case |
|----------|----------|
| `tts` | TTS training data: denoise, segment, ASR, word alignment |
| `asr` | ASR training data: augmentation, transcription, filtering |
| `cleaning` | Data cleaning: quality metrics, dedup, filtering |
| `speaker` | Speaker analysis: diarization, embeddings, gender, language |

See all templates: `vkit init --list-templates`

### Option B: Start from scratch

```bash
vkit init my-project
cd my-project
```

### Add your audio and run

```bash
# Put audio files in the data/ directory
cp /path/to/your/audio/*.wav data/

# Validate the pipeline
vkit run pipeline.yaml --dry-run

# Run it
vkit run pipeline.yaml

# Inspect results
vkit inspect run work/                              # stage summary + timing
vkit inspect cuts work/*/07_pack/cuts.jsonl.gz       # cut statistics
open work/report.html                                # visual report
```

## Download a dataset

Download and process a public dataset in two commands:

```bash
# Download LibriSpeech dev-clean (5.4 hours, ~350 MB)
vkit download librispeech --root ./librispeech --subsets dev-clean

# Process it
vkit init ls-project --template asr
# Edit pipeline.yaml: change ingest root to ./librispeech
vkit run pipeline.yaml
```

Available datasets: `librispeech`, `aishell`, `fleurs`. See [Recipes & Download](reference/recipes.md).

## Discover operators

```bash
# List all 51 operators (grouped by category)
vkit operators

# Show config fields + YAML example for any operator
vkit operators show silero_vad
vkit operators show qwen3_asr
vkit operators show noise_augment
```

## Python tools API

For quick one-off tasks without writing a YAML pipeline:

```python
from voxkitchen.tools import (
    audio_info, transcribe, detect_speech, estimate_snr,
    extract_speaker_embedding, enhance_speech, align_words,
    synthesize,
)

audio_info("speech.wav")
# AudioInfo(sample_rate=16000, duration=3.2, num_channels=1, format='WAV')

transcribe("speech.wav", model="tiny")
# [SpeechSegment(start=0.0, end=3.2, text="Hello world")]

estimate_snr("speech.wav")
# 18.3

# Speaker embedding (requires: pip install voxkitchen[speaker])
emb = extract_speaker_embedding("speaker.wav")

# Forced alignment (requires: pip install voxkitchen[align])
align_words("speech.wav", "hello world", language="English")

# TTS synthesis (requires: pip install voxkitchen[tts-kokoro])
synthesize("Hello world!", "output.wav", engine="kokoro")

# Voice cloning (requires: pip install voxkitchen[tts-cosyvoice])
synthesize("你好", "clone.wav", engine="cosyvoice",
           reference_audio="ref.wav", reference_text="参考文本")
```

## Next steps

- [TTS Tutorial](tutorials/tts-data-prep.md) — end-to-end TTS data preparation
- [ASR Tutorial](tutorials/asr-training-data.md) — augmented ASR training data
- [Data Cleaning Tutorial](tutorials/data-cleaning.md) — quality metrics and filtering
- [Operators Reference](reference/operators.md) — all 51 operators with config details
- [Data Protocol](concepts/data-protocol.md) — understand Recording / Supervision / Cut
