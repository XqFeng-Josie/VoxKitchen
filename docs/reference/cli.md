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

### `vkit run` (container entrypoint)

Execute a pipeline in the current Python environment. This is the
container entrypoint used by VoxKitchen images. Most host users should use
`vkit docker run <yaml>`, which supplies the runtime image and Docker mounts.

```bash
# Host usage; these flags are forwarded to the image entrypoint:
vkit docker run pipeline.yaml                           # Run full pipeline
vkit docker run pipeline.yaml --dry-run                 # Validate only
vkit docker run pipeline.yaml --resume-from vad         # Resume from a stage
vkit docker run pipeline.yaml --stop-at asr             # Stop after a stage
vkit docker run pipeline.yaml --keep-intermediates      # Keep derived audio
vkit docker run pipeline.yaml --num-gpus 2              # Override GPU count
vkit docker run pipeline.yaml --num-workers 8           # Override CPU workers
vkit docker run pipeline.yaml --work-dir ./work/run1    # Override work_dir
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

Check YAML syntax, operator references, per-operator arg schemas, and the
recommended Docker image without executing.

```bash
vkit validate pipeline.yaml
```

### `vkit download` (current-env helper)

Download a dataset using its recipe in the current environment. For the
Docker-first user path, use `vkit docker download` so recipe-specific
dependencies come from the image.

```bash
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean,test-clean
vkit docker download --tag slim aishell --root ./data/aishell
vkit docker download --tag slim fleurs --root ./data/fleurs --subsets en_us,zh_cn
```

| Flag | Meaning |
|------|---------|
| `--root PATH` | Directory to download into (**required**). |
| `--subsets LIST` | Comma-separated subset names. Recipe-specific; see [Recipes & Download](recipes.md). |

### `vkit ingest`

Build a CutSet manifest from a data source — standalone, outside a pipeline.
Most users let `vkit docker run` do this through the pipeline ingest block;
`vkit ingest` is useful for one-off manifest prep.

```bash
vkit ingest --source dir      --root ./data/audio       --out cuts.jsonl.gz
vkit ingest --source recipe   --recipe librispeech      --root ./data/librispeech --out cuts.jsonl.gz
vkit ingest --source manifest --path input.jsonl.gz --out merged.jsonl.gz
vkit ingest --source dir      --root ./data/audio       --out cuts.jsonl.gz --no-recursive
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
vkit operators                       # List all operators (grouped by category)
vkit operators --category quality    # List only operators in one category
vkit operators search noise          # Find operators whose name or description matches "noise"
vkit operators show silero_vad       # Show config fields + YAML example for an operator
```

`search` matches case-insensitively against the operator name and the first
line of its docstring. It exits with code 1 when nothing matches, so shell
scripts can branch on no-result.

Valid `--category` values are: `basic`, `segment`, `augment`, `annotate`,
`quality`, `synthesize`, `pack`, `noop`.

### `vkit schema`

Generate JSON Schemas for YAML editor integration.

```bash
vkit schema export                                  # → ./pipeline.schema.json
vkit schema export --out docs/schemas/pipeline.schema.json   # custom path
```

The output is consumed by YAML language servers (VS Code, Neovim, JetBrains) so
users get autocompletion and inline validation while editing `pipeline.yaml`.
`vkit init` already writes the right `# yaml-language-server: $schema=…`
directive at the top of every scaffolded pipeline. See
[Pipeline JSON Schema](schema.md) for editor setup.

### `vkit recipes`

List dataset recipes (the entities behind `vkit docker download` and `ingest: source=recipe`).

```bash
vkit recipes
```

Output is a table with name, download mechanism (`openslr`, `HuggingFace`, or `manual`), and one-line description. To actually download, use `vkit docker download --tag slim <name> --root ./data/<name>`; to reference inside a pipeline, use `ingest: { source: recipe, recipe: <name>, args: { root: <dir> } }`. Recipe-specific subset names are listed in [Recipes & Download](recipes.md).

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

**Image selection flags** (`run`, `download`, `doctor`, `pull`, `shell`):

| Flag | Default | Meaning |
|------|---------|---------|
| `--tag NAME` | `latest` (`download`: `slim`) | Image tag. Resolves to `ghcr.io/xqfeng-josie/voxkitchen:NAME`. |
| `--image REF` | — | Full image reference; overrides `--tag`. |

#### `vkit docker run <yaml>`

Execute a pipeline inside the container.

```bash
vkit docker run pipeline.yaml                          # :latest
vkit docker run pipeline.yaml --tag asr                # :asr
vkit docker run pipeline.yaml --gpus none              # CPU-only
vkit docker run pipeline.yaml --dry-run                # Validate inside image
vkit docker run pipeline.yaml --env-file /tmp/.env     # Alternate env file
vkit docker run pipeline.yaml --mount /data/raw        # Extra read-only bind mount
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--gpus MODE` | `auto` | `auto` (attach all GPUs if `nvidia-smi` is on PATH), `all`, or `none`. |
| `--env-file PATH` | `./.env` if present | `docker --env-file` path (used for `HF_TOKEN`). |
| `--mount PATH`, `-m` | — | Extra host path to bind read-only. Repeatable. |
| `--dry-run`, `--resume-from`, `--stop-at`, `--num-gpus`, `--num-workers`, `--work-dir`, `--keep-intermediates` | — | Pipeline options forwarded to the image entrypoint. |

The wrapper automatically:

- Sets `--user $(id -u):$(id -g)` and `-e HOME=/tmp` so files in `./work` are owned by the host user.
- Sets `NUMBA_CACHE_DIR=/app/work/.numba-cache` so librosa/numba operators can cache under the mounted work directory.
- Binds `./work → /app/work` and `./output → /app/output`; if `./data` exists, binds it to both `/app/data` for template-relative YAML and `/data` for absolute data roots.
- Binds the pipeline YAML at its absolute path when it points to a host file.

#### `vkit docker doctor`

Run `vkit doctor` inside the container.

```bash
vkit docker doctor                                    # :latest, multi-env aggregate
vkit docker doctor --tag slim                         # slim image, single-env
vkit docker doctor --tag asr --expect asr --json      # smoke test + JSON
```

Accepts `--expect` and `--json` (same semantics as local `vkit doctor`). Default `--gpus` is `none` (doctor doesn't need GPU).

#### `vkit docker download <recipe>`

Download a dataset inside the container. The wrapper creates and mounts
`./data`, so roots under `./data/...` are written back to the host.

```bash
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean
vkit docker download --tag slim fleurs --root ./data/fleurs --subsets en_us,zh_cn
```

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

By default the wrapper keeps Docker client temp/config/cache files under
`./.docker` (`DOCKER_CONFIG`, `TMPDIR`, `BUILDX_CONFIG`, `XDG_CACHE_HOME`).
Set `VKIT_DOCKER_WORK_DIR=/path/to/.docker` to choose a different base
directory. Docker image layers still live under the Docker daemon's
`data-root` (often `/var/lib/docker`); move that daemon setting separately if
`/` is full.

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

`vkit viz` is an optional local developer UI; it is separate from the
Docker-first pipeline execution path.
