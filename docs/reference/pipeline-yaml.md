# Pipeline YAML Reference

A VoxKitchen pipeline is defined as a YAML file with three sections: metadata, ingest, and stages.

## Full Schema

```yaml
version: "0.1"                          # Required. Schema version.
name: my-pipeline                       # Required. Pipeline name.
description: "What this pipeline does"  # Optional.

work_dir: ./work/${name}-${run_id}      # Required. Output directory.
num_gpus: 1                             # Optional. GPU count (default: 1).
num_cpu_workers: null                   # Optional. CPU workers (default: auto).
gc_mode: aggressive                     # Optional. "aggressive" or "keep".

ingest:                                 # Required. How data enters the pipeline.
  source: dir | manifest | recipe       # Required. Ingest source type.
  recipe: librispeech                   # For source=recipe only.
  args:                                 # Source-specific arguments.
    root: /path/to/data
    recursive: true

stages:                                 # Required. Processing steps.
  - name: my_stage                      # Required. Unique stage name.
    op: operator_name                   # Required. Registered operator name.
    args:                               # Optional. Operator-specific config.
      param1: value1
      param2: value2
```

## String Interpolation

Pipeline YAML supports variable substitution:

| Variable | Value |
|----------|-------|
| `${name}` | Pipeline name |
| `${run_id}` | Generated run ID (e.g., `run-20260415-a1b2c3`) |
| `${env:VAR_NAME}` | Environment variable |

```yaml
work_dir: ./work/${name}-${run_id}      # → ./work/my-pipeline-run-20260415-a1b2c3
```

## Ingest Sources

### `dir` — Scan a directory for audio files

```yaml
ingest:
  source: dir
  args:
    root: /path/to/audio          # Required.
    recursive: true               # Optional (default: true).
```

### `manifest` — Load a pre-built CutSet

```yaml
ingest:
  source: manifest
  args:
    path: /path/to/cuts.jsonl.gz  # Required.
```

### `recipe` — Use a dataset recipe

```yaml
ingest:
  source: recipe
  recipe: librispeech             # Recipe name.
  args:
    root: /path/to/librispeech    # Required.
    subsets: [train-clean-100]    # Optional. Default: all subsets.
```

Available recipes: `librispeech`, `aishell`, `commonvoice`, `fleurs`

## Pipeline Execution

### Resume

Pipelines checkpoint after each stage. If a run crashes, re-running resumes from the last completed stage:

```bash
vkit run pipeline.yaml                          # Auto-resume
vkit run pipeline.yaml --resume-from asr        # Force resume from specific stage
```

### Partial Execution

```bash
vkit run pipeline.yaml --stop-at vad            # Stop after VAD stage
```

### Garbage Collection

By default (`gc_mode: aggressive`), intermediate audio files are cleaned up after downstream stages finish. Use `--keep-intermediates` to preserve all derived audio.

## Stage Execution

Stages execute sequentially. Each stage:

1. Receives the CutSet from the previous stage
2. Splits it across CPU/GPU workers
3. Runs the operator on each shard
4. Merges results and writes `cuts.jsonl.gz` + `_SUCCESS` marker + `_stats.json`

Failed cuts are logged to `_errors.jsonl` and skipped — the pipeline continues.
