# CLI Reference

VoxKitchen provides the `vkit` command-line tool.

## Commands

### `vkit init`

Scaffold a new pipeline project.

```bash
vkit init my-project                        # Empty template
vkit init my-project --template tts         # TTS data preparation
vkit init my-project --template asr         # ASR training data
vkit init my-project --template cleaning    # Data cleaning
vkit init my-project --template speaker     # Speaker analysis
vkit init --list-templates                  # Show available templates
```

| Flag | Meaning |
|------|---------|
| `--template`, `-t` | Project template (`tts`, `asr`, `cleaning`, `speaker`). |
| `--list-templates` | Print templates and exit. |

### `vkit run`

Execute a pipeline.

```bash
vkit run pipeline.yaml                           # Run full pipeline
vkit run pipeline.yaml --dry-run                 # Validate only (no execution)
vkit run pipeline.yaml --resume-from vad         # Resume from a stage
vkit run pipeline.yaml --stop-at asr             # Stop after a stage
vkit run pipeline.yaml --keep-intermediates      # Don't GC derived audio
vkit run pipeline.yaml --num-gpus 2              # Override GPU count
vkit run pipeline.yaml --num-workers 8           # Override CPU workers
vkit run pipeline.yaml --work-dir /tmp/run1      # Override work_dir
```

| Flag | Meaning |
|------|---------|
| `--dry-run` | Parse + validate the pipeline, resolve the stage plan, exit without executing. |
| `--resume-from STAGE` | Force-resume from `STAGE` regardless of existing checkpoints. |
| `--stop-at STAGE` | Stop after `STAGE` completes. |
| `--keep-intermediates` | Disable GC; keep every stage's derived audio on disk. |
| `--num-gpus N` | Override the pipeline YAML's `num_gpus`. |
| `--num-workers N` | Override `num_cpu_workers`. |
| `--work-dir PATH` | Override the pipeline YAML's `work_dir`. |

### `vkit validate`

Check YAML syntax, operator references, and per-operator arg schemas without executing.

```bash
vkit validate pipeline.yaml
```

### `vkit download`

Download a dataset using its recipe.

```bash
vkit download librispeech --root /data/ls --subsets dev-clean,test-clean
vkit download aishell --root /data/aishell
vkit download fleurs --root /data/fleurs --subsets en_us,zh_cn
```

| Flag | Meaning |
|------|---------|
| `--root PATH` | Directory to download into (**required**). |
| `--subsets LIST` | Comma-separated subset names. Recipe-specific; see [Recipes & Download](recipes.md). |

### `vkit ingest`

Build a CutSet manifest from a data source — standalone, outside a pipeline. Most users let `vkit run` do this as stage 0; `vkit ingest` is useful for one-off manifest prep.

```bash
vkit ingest --source dir      --root /data/audio    --out cuts.jsonl.gz
vkit ingest --source recipe   --recipe librispeech  --root /data/ls --out cuts.jsonl.gz
vkit ingest --source manifest --path input.jsonl.gz --out merged.jsonl.gz
vkit ingest --source dir      --root /data/audio    --out cuts.jsonl.gz --no-recursive
```

| Flag | Source | Meaning |
|------|--------|---------|
| `--source` | all | `dir`, `manifest`, or `recipe`. |
| `--out` | all | Output `cuts.jsonl.gz` path (**required**). |
| `--root` | `dir`, `recipe` | Dataset root directory. |
| `--path` | `manifest` | Path to an input `cuts.jsonl.gz`. |
| `--recipe` | `recipe` | Recipe name (`librispeech`, `aishell`, `commonvoice`, `fleurs`). |
| `--subsets` | `recipe` | Comma-separated subset names. |
| `--recursive / --no-recursive` | `dir` | Recurse into subdirectories (default: recurse). |

### `vkit inspect`

Inspect pipeline results and cut data. Four subcommands:

```bash
vkit inspect cuts   work/01_pack/cuts.jsonl.gz    # CutSet statistics
vkit inspect run    work/                          # Per-stage summary + timing
vkit inspect trace  utt-001 --in work/             # Provenance chain for one cut
vkit inspect errors work/                          # Per-stage error entries
```

| Subcommand | Argument | Purpose |
|------------|----------|---------|
| `cuts <path>` | A `cuts.jsonl.gz` | Print CutSet-level stats (duration, sample rate, metric histograms). |
| `run <work_dir>` | Pipeline work dir | Table of stage name, cut count, duration, success marker. |
| `trace <cut_id> --in <work_dir>` | Cut id + work dir | Walk `Provenance` parent links to show where a cut came from. |
| `errors <work_dir>` | Pipeline work dir | Dump per-stage `_errors.jsonl` entries (cuts that failed). |

### `vkit operators`

List and inspect operators.

```bash
vkit operators                  # List all operators (grouped by category)
vkit operators show silero_vad  # Show config fields + YAML example for an operator
```

### `vkit recipes`

List dataset recipes (the entities behind `vkit download` and `ingest: source=recipe`).

```bash
vkit recipes
```

Output is a table with name, download mechanism (`openslr`, `HuggingFace`, or `manual`), and one-line description. To actually download, use `vkit download <name> --root <dir>`; to reference inside a pipeline, use `ingest: { source: recipe, recipe: <name>, args: { root: <dir> } }`. Recipe-specific subset names are listed in [Recipes & Download](recipes.md).

### `vkit doctor`

Report per-env operator availability and warmup-model status.

```bash
vkit doctor                          # Single-env report (dev install or :slim image)
vkit doctor --expect core            # Assert core-env expected operators are importable
vkit doctor --expect asr             # Asserts for :asr image
vkit doctor --json                   # Machine-readable output on stdout
```

| Flag | Meaning |
|------|---------|
| `--expect ENV` | Image env to validate against (`core`, `asr`, `diarize`, `tts`, `fish-speech`). Exits non-zero if any expected operator fails to import. Used by the Dockerfile's per-stage smoke test. |
| `--json` | Emit a JSON report on stdout (rich table still goes to stderr). |

Inside the `voxkitchen:latest` multi-env image, `vkit doctor` with no `--expect` aggregates a table across every env under `/opt/voxkitchen/envs/`, re-invoking each env's own `vkit doctor --expect <env>`.

### `vkit docker`

Run any of the commands above inside a published Docker image instead of the local Python env. Also has three image-management helpers (`build`, `pull`, `shell`).

**Shared flags** (every subcommand):

| Flag | Default | Meaning |
|------|---------|---------|
| `--tag NAME` | `latest` | Image tag. Resolves to `ghcr.io/xqfeng-josie/voxkitchen:NAME`. |
| `--image REF` | — | Full image reference; overrides `--tag`. |

#### `vkit docker run <yaml>`

Execute a pipeline inside the container.

```bash
vkit docker run pipeline.yaml                          # :latest
vkit docker run pipeline.yaml --tag asr                # :asr
vkit docker run pipeline.yaml --gpus none              # CPU-only
vkit docker run pipeline.yaml --env-file /tmp/.env     # Alternate env file
vkit docker run pipeline.yaml --mount /data/raw        # Extra read-only bind mount
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--gpus MODE` | `auto` | `auto` (attach all GPUs if `nvidia-smi` is on PATH), `all`, or `none`. |
| `--env-file PATH` | `./.env` if present | `docker --env-file` path (used for `HF_TOKEN`). |
| `--mount PATH`, `-m` | — | Extra host path to bind read-only. Repeatable. |

The wrapper automatically:

- Sets `--user $(id -u):$(id -g)` and `-e HOME=/tmp` so files in `./work` are owned by the host user.
- Binds `./work → /app/work` and `./data → /data` if they exist.
- Binds the pipeline YAML at its absolute path when it points to a host file.

#### `vkit docker doctor`

Run `vkit doctor` inside the container.

```bash
vkit docker doctor                                    # :latest, multi-env aggregate
vkit docker doctor --tag slim                         # slim image, single-env
vkit docker doctor --tag asr --expect asr --json      # smoke test + JSON
```

Accepts `--expect` and `--json` (same semantics as local `vkit doctor`). Default `--gpus` is `none` (doctor doesn't need GPU).

#### `vkit docker build [target]`

Build a local Docker image from `docker/Dockerfile` (wraps `docker build`).

```bash
vkit docker build                 # Default target: latest
vkit docker build slim
vkit docker build asr
vkit docker build latest --tag voxkitchen:dev
vkit docker build latest --no-hf-token                # Skip baking pyannote
```

| Argument/Flag | Default | Meaning |
|---------------|---------|---------|
| `target` | `latest` | Dockerfile target: `slim`, `asr`, `diarize`, `tts`, `fish-speech`, `latest`. |
| `--tag NAME` | `voxkitchen:<target>` | Image tag to apply. |
| `--hf-token / --no-hf-token` | `--hf-token` | Pass `HF_TOKEN` from `./.env` as a build arg so pyannote is baked into the image. |

Pass extra `docker build` flags after `--`:

```bash
vkit docker build latest -- --no-cache --progress=plain
```

#### `vkit docker pull`

Pull a published image from GHCR.

```bash
vkit docker pull                      # :latest
vkit docker pull --tag slim
vkit docker pull --image my-registry/vox:custom
```

#### `vkit docker shell`

Drop into an interactive `bash` inside the image, useful for debugging.

```bash
vkit docker shell --tag slim
vkit docker shell --tag latest --gpus all
```

### `vkit viz`

Launch an interactive Gradio panel to explore a CutSet.

```bash
vkit viz work/01_pack/cuts.jsonl.gz --port 7860
```

Requires: `pip install voxkitchen[viz-panel]`
