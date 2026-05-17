<p align="center">
  <img src="voxkitchen_logo.svg" width="400" alt="VoxKitchen logo">
</p>

<p align="center">
  <a href="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml"><img src="https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

Declarative speech data processing toolkit. Write a YAML recipe, run it with `vkit docker`, get training-ready data.

> **Status:** Pre-alpha. API is unstable.

## Quick Start

VoxKitchen runs user pipelines in prebuilt Docker images. The local `vkit`
command is a lightweight launcher: it handles Docker flags, mounts, `.env`
loading, GPU autodetection, and file ownership for you.

Requirements:

- Docker
- Python 3.10+ for the lightweight `vkit` CLI

Install the CLI:

```bash
export VKIT_VERSION=v0.2.0

python -m venv ~/.venvs/voxkitchen
~/.venvs/voxkitchen/bin/python -m pip install -U pip
~/.venvs/voxkitchen/bin/python -m pip install \
  "voxkitchen @ https://github.com/XqFeng-Josie/VoxKitchen/archive/refs/tags/${VKIT_VERSION}.zip"

mkdir -p ~/.local/bin
ln -sf ~/.venvs/voxkitchen/bin/vkit ~/.local/bin/vkit
export PATH="$HOME/.local/bin:$PATH"
```

This installs only the lightweight launcher and inspection commands. Pipeline
dependencies stay inside Docker images.

If `vkit` is not found in a new shell, add the `PATH` line above to your
shell startup file, such as `~/.bashrc` or `~/.zshrc`.

Pull a prebuilt runtime image and run the demo. No repository clone is required
for this quick start; the published image includes the demo pipeline and demo
audio.

```bash
vkit docker pull --tag slim
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
vkit inspect run ./work/demo-no-asr
```

`vkit docker run` writes run artifacts under `./work` and exported datasets
under `./output` with your host user ID. It also mounts `./data`
automatically when that directory exists.

## Create A Project

Scaffold a project, put audio under `data/`, check the plan, then run it
inside a prebuilt image:

```bash
vkit init my-project --template asr
cd my-project

# Put your audio files in ./data first.
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/
```

Templates:

| Template | Use case | Suggested image |
|---|---|---|
| `cleaning` | Quality metrics, dedup, filtering | `slim` |
| `asr` | VAD, augmentation, ASR labeling, packing | `asr` |
| `speaker` | VAD, diarization, speaker embedding, language/gender labels | `latest` |
| `tts` | Denoise, segment, transcribe, align, pack | `latest` |

List templates:

```bash
vkit init --list-templates
```

## How It Works

A pipeline is a YAML file: ingest raw audio, pass it through stages, and
write the result out.

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${name}-${run_id}

ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }

  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }

  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, compute_type: float16 }

  - name: filter
    op: quality_score_filter
    args:
      conditions: ["duration > 1", "duration < 30", "metrics.snr > 10"]

  - name: pack
    op: pack_jsonl
```

VoxKitchen checkpoints every stage under `work_dir`, so interrupted runs
can resume from completed stages.

## Prebuilt Images

Every `vkit docker` command accepts `--tag <name>`:

| Tag | Use when | GPU | Approx. size |
|---|---|---|---|
| `slim` | CPU-friendly cleaning, VAD, quality, pack, enhancement | no | ~13 GB |
| `asr` | Faster-Whisper, FunASR, Qwen3-ASR, forced alignment | yes | ~48 GB |
| `diarize` | Pyannote speaker diarization | yes | ~32 GB |
| `tts` | Kokoro, ChatTTS, CosyVoice | yes | ~44 GB |
| `fish-speech` | Fish-Speech isolated runtime | yes | ~57 GB |
| `latest` | Mixed pipelines across ASR, diarization, TTS, or Fish-Speech | yes | ~123 GB |

Examples:

```bash
vkit docker run --tag slim pipeline.yaml
vkit docker run --tag asr pipeline.yaml
vkit docker doctor --tag latest
```

Use `latest` when a pipeline mixes operators from multiple runtime
families, such as diarization plus ASR or TTS plus ASR.

Not sure which image a pipeline needs? Run:

```bash
vkit validate pipeline.yaml
```

It prints the recommended `vkit docker pull --tag ...` and
`vkit docker run --tag ...` commands for that YAML.

## Configuration

Some operators require API tokens. Create `./.env`; `vkit docker run`
passes it into the container automatically.

```bash
cp .env.example .env
```

| Variable | Required by | Notes |
|---|---|---|
| `HF_TOKEN` | `pyannote_diarize` | Accept the pyannote model agreement on HuggingFace first. |

## Common Commands

```bash
vkit init <path> --template asr       # Scaffold a project
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/                # Stage summary
vkit inspect cuts <cuts.jsonl.gz>      # CutSet statistics
vkit inspect errors work/              # Per-stage failed cuts
vkit recipes                           # List dataset recipes
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean
vkit docker doctor --tag latest        # Check image health
```

Full CLI reference: [docs/reference/cli.md](docs/reference/cli.md).

## Operators

51 built-in operators across audio processing, segmentation,
augmentation, annotation, quality metrics, TTS synthesis, and output
packing.

Full operator reference: [docs/reference/operators.md](docs/reference/operators.md).

## Documentation

- [Getting Started](docs/getting-started.md)
- [Pipeline YAML](docs/reference/pipeline-yaml.md)
- [Recipes & Download](docs/reference/recipes.md)
- [CLI reference](docs/reference/cli.md)
- [Operators reference](docs/reference/operators.md)

Contributor setup and project internals are covered in
[Contributing](CONTRIBUTING.md).

Clone the repository only when you want to browse or edit examples, use the
bundled `skill/`, contribute code, or build images locally:

```bash
git clone https://github.com/XqFeng-Josie/VoxKitchen.git
cd VoxKitchen
```

## Key Features

- **Docker-first execution** — prebuilt runtimes avoid local dependency conflicts
- **Resumable** — each stage checkpoints; crashes resume from the last good stage
- **Inspectable** — per-stage summaries, cut statistics, errors, and provenance
- **Error tolerant** — bad cuts are logged to `_errors.jsonl` and skipped
- **Extensible** — custom operators can be registered through Python entry points

## Examples

The published Docker images include ready-to-run demo YAML files. Clone the
repository if you want to inspect or modify the full
[`examples/pipelines/`](examples/pipelines/) set locally.

## Agent Skill

The repo includes an agent-neutral VoxKitchen skill at [skill/](skill/). Claude,
Codex, and other `SKILL.md`-compatible agents can copy, symlink, or import that
folder into their own skill search path. The skill follows the Docker-first
`vkit` workflow in this README.

## License

Apache 2.0. See [LICENSE](LICENSE).
