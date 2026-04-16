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

### `vkit run`

Execute a pipeline.

```bash
vkit run pipeline.yaml                              # Run full pipeline
vkit run pipeline.yaml --dry-run                     # Validate only
vkit run pipeline.yaml --resume-from vad             # Resume from a stage
vkit run pipeline.yaml --stop-at asr                 # Stop after a stage
vkit run pipeline.yaml --keep-intermediates           # Don't clean up derived audio
vkit run pipeline.yaml --num-gpus 2                  # Override GPU count
vkit run pipeline.yaml --num-workers 8               # Override CPU workers
```

### `vkit validate`

Check YAML syntax and operator references without executing.

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

### `vkit ingest`

Build a CutSet manifest from a data source (standalone, outside pipeline).

```bash
vkit ingest --source dir --root /data/audio --out cuts.jsonl.gz
vkit ingest --source recipe --recipe librispeech --root /data/ls --out cuts.jsonl.gz
vkit ingest --source manifest --path input.jsonl.gz --out output.jsonl.gz
```

### `vkit inspect`

Inspect pipeline results and data.

```bash
vkit inspect cuts work/01_pack/cuts.jsonl.gz    # CutSet statistics
vkit inspect run work/                           # Stage summary with timing
vkit inspect trace <cut_id> --in work/           # Trace a cut's provenance
vkit inspect errors work/                        # Show per-stage errors
```

### `vkit operators`

List and inspect operators.

```bash
vkit operators                  # List all operators (grouped by category)
vkit operators show silero_vad  # Show config + YAML example for an operator
```

### `vkit viz`

Launch an interactive Gradio panel to explore a CutSet.

```bash
vkit viz work/01_pack/cuts.jsonl.gz --port 7860
```

Requires: `pip install voxkitchen[viz-panel]`
