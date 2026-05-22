# VoxKitchen CLI Reference For Agents

Use this when an agent needs exact command patterns. The user-facing execution
model is Docker-first: local `vkit` launches and inspects; pipelines and dataset
downloads run inside prebuilt images.

## First Run

```bash
pipx install voxkitchen      # or: pip install voxkitchen
vkit docker pull --tag slim
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml --dry-run
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
vkit inspect run ./work/demo-no-asr
```

No repository clone is required for this quick start; the published Docker
image includes the demo pipeline and demo audio.

## Create And Run A Project

```bash
vkit init my-project --template asr
cd my-project
cp /path/to/audio/*.wav data/

vkit validate pipeline.yaml
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/
```

Templates:

| Template | Suggested image |
|---|---|
| `cleaning` | `slim` |
| `asr` | `asr` |
| `speaker` | `latest` |
| `tts` | `latest` |

## Image Selection

Prefer the automatic recommendation:

```bash
vkit validate pipeline.yaml
```

It prints:

```text
recommended image: <tag>
  pull: vkit docker pull --tag <tag>
  run:  vkit docker run --tag <tag> pipeline.yaml
```

General tag meanings:

| Tag | Use when |
|---|---|
| `slim` | CPU-friendly cleaning, VAD, quality, pack, enhancement |
| `asr` | Faster-Whisper, FunASR, Qwen3-ASR, WeNet, forced alignment |
| `diarize` | Pyannote speaker diarization |
| `tts` | Kokoro, ChatTTS, CosyVoice |
| `fish-speech` | Fish-Speech |
| `latest` | Mixed specialized runtime families |

## Pipeline Execution

Use the Docker wrapper:

```bash
vkit docker run --tag <tag> pipeline.yaml
vkit docker run --tag <tag> pipeline.yaml --dry-run
vkit docker run --tag <tag> pipeline.yaml --resume-from <stage>
vkit docker run --tag <tag> pipeline.yaml --stop-at <stage>
vkit docker run --tag <tag> pipeline.yaml --num-gpus 2
vkit docker run --tag <tag> pipeline.yaml --num-workers 8
vkit docker run --tag <tag> pipeline.yaml --work-dir ./work/run1
vkit docker run --tag <tag> pipeline.yaml --keep-intermediates
```

The wrapper mounts `./work`, `./output`, and existing `./data`; it also loads
`./.env` when present and writes files as the host user.

## Dataset Recipes

Run downloads in Docker:

```bash
vkit docker download librispeech --root ./data/librispeech --subsets dev-clean
vkit docker download libritts --root ./data/libritts --subsets dev-clean
vkit docker download ljspeech --root ./data/ljspeech
vkit docker download aishell --root ./data/aishell
vkit docker download aishell3 --root ./data/aishell3
vkit docker download cnceleb --root ./data/cnceleb
vkit docker download fleurs --root ./data/fleurs --subsets en_us,zh_cn
vkit docker download musan --root ./data/musan
# commonvoice: manual download (see docs/reference/recipes.md)
vkit recipes
```

Reference downloaded data in YAML:

```yaml
ingest:
  source: recipe
  recipe: librispeech
  args:
    root: ./data/librispeech
    subsets: [dev-clean]
```

## Inspection

```bash
vkit inspect run work/
vkit inspect cuts <work_dir>/<stage>/cuts.jsonl.gz
vkit inspect errors work/
vkit inspect trace <cut-id> --in work/
```

## Discovery

```bash
vkit init --list-templates
vkit operators
vkit operators show silero_vad
vkit operators show quality_score_filter
vkit recipes
vkit docker doctor --tag latest
```

## Developer-Only Commands

Use these only when the user explicitly asks for contributor or image
maintenance work:

```bash
vkit run pipeline.yaml
vkit download <recipe> --root <dir>
vkit docker build <target>
vkit docker shell --tag <tag>
```
