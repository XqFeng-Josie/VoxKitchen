# VoxKitchen

Declarative speech data processing toolkit. Write a YAML recipe, run `vkit run`, get training-ready data.

**51 operators** across 7 categories: audio processing, segmentation, augmentation, annotation (ASR/diarization/alignment/emotion), quality metrics, TTS synthesis, and output packing.

## Get Started

- [Getting Started](getting-started.md) — install, first pipeline, inspect results
- [Data Protocol](concepts/data-protocol.md) — Recording, Supervision, Cut, CutSet, Provenance

## Tutorials

Use a template to scaffold a project for your use case:

```bash
vkit init my-project --template tts       # TTS data preparation
vkit init my-project --template asr       # ASR training data
vkit init my-project --template cleaning  # Data cleaning
vkit init my-project --template speaker   # Speaker analysis
```

- [TTS Data Preparation](tutorials/tts-data-prep.md) — denoise, segment, transcribe, align
- [ASR Training Data](tutorials/asr-training-data.md) — augment, transcribe, pack for training
- [Data Cleaning](tutorials/data-cleaning.md) — quality metrics, dedup, filter
- [Speaker Analysis](tutorials/speaker-analysis.md) — diarize, embed, classify

## Reference

- [Operators](reference/operators.md) — all 51 operators with config and YAML examples
- [Recipes & Download](reference/recipes.md) — dataset recipes and `vkit download`
- [CLI Commands](reference/cli.md) — complete CLI reference
- [Python Tools API](reference/tools-api.md) — standalone functions for quick tasks
- [Pipeline YAML](reference/pipeline-yaml.md) — YAML schema and execution model

## Quick Reference

```bash
vkit operators                      # list all operators
vkit operators show <name>          # config fields + YAML example
vkit init --list-templates          # available project templates
vkit download librispeech --root /data/ls --subsets dev-clean
vkit run pipeline.yaml --dry-run    # validate without executing
```

## Links

- [Example pipelines](https://github.com/voxkitchen/voxkitchen/tree/main/examples/pipelines)
- [GitHub](https://github.com/voxkitchen/voxkitchen)
- [License](https://github.com/voxkitchen/voxkitchen/blob/main/LICENSE) (Apache 2.0)
