# VoxKitchen CLI Reference (`vkit`)

Entry point: `vkit` (installed via `pip install -e ".[...]"`)

---

## Project Scaffolding

```bash
vkit init <path> [--template <tts|asr|cleaning|speaker>]
vkit init --list-templates
```

Creates:
```
my-project/
├── pipeline.yaml
├── data/
└── README.md
```

Templates:
- `tts` — TTS data prep (denoise → vad → quality → asr → alignment)
- `asr` — ASR training data (vad → augmentation → asr → filter)
- `cleaning` — Data cleaning (quality metrics → dedup → filter)
- `speaker` — Speaker analysis (vad → diarize → embeddings → gender)

---

## Pipeline Execution

```bash
vkit run <yaml>                              # Execute full pipeline
vkit run <yaml> --dry-run                    # Validate only, no execution
vkit run <yaml> --resume-from <stage_name>   # Force resume from stage
vkit run <yaml> --stop-at <stage_name>       # Stop after this stage
vkit run <yaml> --num-gpus 2                 # Override GPU count
vkit run <yaml> --num-workers 8              # Override CPU worker count
vkit run <yaml> --work-dir /tmp/run1         # Override output directory
vkit run <yaml> --keep-intermediates         # Don't delete intermediate audio
```

---

## Validation & Diagnostics

```bash
vkit validate <yaml>                    # Parse + validate YAML + check operator args
vkit doctor                             # Per-env operator availability + model cache
vkit doctor --expect core               # Assert core-env operators importable
vkit doctor --expect asr                # Assert asr-env operators importable
vkit doctor --json                      # Machine-readable output
```

---

## Data Ingestion (standalone, outside pipeline)

```bash
vkit ingest --source dir      --root /data/audio    --out cuts.jsonl.gz
vkit ingest --source dir      --root /data/audio    --out cuts.jsonl.gz --no-recursive
vkit ingest --source recipe   --recipe librispeech  --root /data/ls --out cuts.jsonl.gz
vkit ingest --source manifest --path input.jsonl.gz --out merged.jsonl.gz
```

---

## Dataset Download

```bash
vkit download <recipe> --root <dir> [--subsets <list>]
```

Available recipes: `librispeech`, `aishell`, `commonvoice`, `fleurs`

```bash
vkit download librispeech --root ./librispeech --subsets dev-clean,test-clean
vkit download aishell --root ./aishell
vkit download fleurs --root ./fleurs --subsets en_us,zh_cn
```

---

## Operator Discovery

```bash
vkit operators                          # List all 51 operators grouped by category
vkit operators show <name>              # Config fields + YAML example for an operator
vkit operators show silero_vad
vkit operators show quality_score_filter
```

---

## Inspection & Analysis

```bash
# Pipeline run summary
vkit inspect run <work_dir>             # Per-stage: input/output counts, errors, timing

# CutSet statistics
vkit inspect cuts <path/to/cuts.jsonl.gz>

# Provenance tracing
vkit inspect trace <cut_id> --in <work_dir>

# Error examination
vkit inspect errors <work_dir>          # All failed cuts across stages
```

---

## Interactive Visualization

```bash
vkit viz <cuts.jsonl.gz> [--port 7860]
```

Requires: `pip install voxkitchen[viz-panel]`
Launches a Gradio panel for exploring the CutSet interactively (audio playback, waveform, metadata).

---

## Dataset Recipe Listing

```bash
vkit recipes                            # List available dataset recipes with descriptions
```

---

## Docker Backend

Prefix any command with `docker` to run in a container:

```bash
# Running pipelines
vkit docker run <yaml>                           # Execute in :latest image
vkit docker run <yaml> --tag asr                 # Use specific image tag
vkit docker run <yaml> --image my.reg/vox:x      # Full image URL override
vkit docker run <yaml> --gpus none               # CPU-only
vkit docker run <yaml> --gpus all                # All GPUs
vkit docker run <yaml> --env-file /tmp/.env      # Alternate env file
vkit docker run <yaml> --mount /data/raw         # Extra read-only mount (repeatable)

# Diagnostics
vkit docker doctor [--tag slim] [--expect asr] [--json]

# Image management
vkit docker build [target]    # slim|asr|diarize|tts|fish-speech|latest
vkit docker pull [--tag slim] # Pull from GHCR

# Interactive shell
vkit docker shell [--tag latest] [--gpus all]
```

Docker image tags:
| Tag | Size | Contains |
|---|---|---|
| `:slim` | ~13 GB | core env (CPU only) |
| `:asr` | ~48 GB | core + asr |
| `:diarize` | ~32 GB | core + diarize |
| `:tts` | ~44 GB | core + tts |
| `:fish-speech` | ~38 GB | core + fish-speech (torch 2.8) |
| `:latest` | ~103 GB | all five envs |

Docker auto-behaviors:
- Detects and mounts GPUs (`--gpus auto` by default)
- Sets `--user $(id -u):$(id -g)` so output files are owned by host user
- Auto-binds `./work` and `./data` directories if they exist
- Loads `.env` automatically (for HF_TOKEN)
- Mounts pipeline YAML at its absolute path
