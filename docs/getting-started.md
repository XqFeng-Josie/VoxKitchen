# Getting Started

VoxKitchen's recommended user path is Docker-first: install the lightweight
`vkit` launcher locally, then run pipelines inside prebuilt Docker images.
You do not need to install model dependencies on your host machine.

## Install The Launcher

Requirements:

- Docker
- Python 3.10+ for the lightweight `vkit` CLI

```bash
pipx install voxkitchen      # recommended — isolates the launcher
# or
pip install voxkitchen
```

This installs only the lightweight launcher and inspection commands (a few MB,
no torch / ASR / TTS dependencies). All pipeline runtime dependencies stay
inside the prebuilt Docker images.

## Pull A Runtime Image

Start with the `slim` image for the demo. For your own pipelines, pick the
smallest image that contains the operators you use. Mixed pipelines may need
`latest`.

```bash
vkit docker pull --tag slim
```

| Tag | Use when |
|---|---|
| `slim` | CPU-friendly cleaning, VAD, quality metrics, packing |
| `asr` | Faster-Whisper, FunASR, Qwen3-ASR, forced alignment |
| `diarize` | Pyannote speaker diarization |
| `tts` | Kokoro, ChatTTS, CosyVoice |
| `fish-speech` | Fish-Speech isolated runtime |
| `latest` | Mixed pipelines across ASR, diarization, TTS, Fish-Speech, and core operators |

Not sure which image your YAML needs? Run `vkit validate pipeline.yaml`; it
prints the recommended `vkit docker pull --tag ...` and run command.

Command flags and tag behavior are listed in the [CLI reference](reference/cli.md).

## Run The Demo

The published image includes example pipelines and demo audio, so no
repository clone is required for this quick start. Start with the `slim` demo;
use `latest` later for pipelines that mix ASR, diarization, and TTS operators.

```bash
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml --dry-run
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
vkit inspect run ./work/demo-no-asr
```

## Create Your First Project

Use a template, put audio under `data/`, validate the plan, then run in
Docker.

```bash
vkit init my-project --template asr
cd my-project

cp /path/to/your/audio/*.wav data/

vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/
```

Available templates:

| Template | Use case | Suggested image |
|---|---|---|
| `cleaning` | Quality metrics, dedup, filtering | `slim` |
| `asr` | VAD, augmentation, ASR labeling, packing | `asr` |
| `speaker` | Diarization, embeddings, language/gender labels | `latest` |
| `tts` | Denoise, segment, transcribe, align, pack | `asr` |

See all templates:

```bash
vkit init --list-templates
```

## Inspect Results

```bash
vkit inspect run work/
vkit inspect cuts <work_dir>/<stage>/cuts.jsonl.gz
vkit inspect errors work/
```

`vkit docker run` writes run artifacts under `./work` and exported datasets
under `./output` with your host user ID. It also mounts `./data`
automatically when that directory exists.

## Download A Dataset

Dataset download also runs through Docker, so recipe dependencies stay inside
the runtime image and data lands under your project's `./data` directory.

```bash
vkit init ls-project --template asr
cd ls-project
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean
# Edit pipeline.yaml: set ingest.args.root to ./data/librispeech
vkit docker run --tag asr pipeline.yaml
```

Run `vkit recipes` to list every available dataset along with its
compressed download size — 9 recipes ship today, covering English /
Chinese ASR (`librispeech`, `aishell`), multi-speaker TTS
(`libritts`, `ljspeech`, `aishell3`), Chinese speaker recognition
(`cnceleb`), multilingual eval (`fleurs`), augmentation (`musan`),
plus `commonvoice` (manual download). See
[Dataset Catalog](datasets/index.md) for per-recipe subset
details.

## Configuration

Some operators require API tokens. Put them in `./.env`; `vkit docker run`
passes that file into the container automatically.

```bash
cp .env.example .env
```

| Variable | Required by | Notes |
|---|---|---|
| `HF_TOKEN` | `pyannote_diarize` | Accept the pyannote model agreement on HuggingFace first. |

## Next Steps

- [TTS Training Data](tutorials/tts-training-data.md)
- [Speaker TTS](tutorials/tts-speaker.md) · [Voice Cloning & TTS](tutorials/tts-voice-cloning.md)
- [ASR Tutorial](tutorials/asr-training-data.md)
- [Data Cleaning Tutorial](tutorials/data-cleaning.md)
- [Operators Reference](reference/operators.md)
- [Pipeline YAML](reference/pipeline-yaml.md)
- [Data Protocol](concepts/data-protocol.md)
