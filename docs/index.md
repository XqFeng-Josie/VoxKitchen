# VoxKitchen

A researcher-friendly, declarative speech data processing toolkit with a unified data protocol.

## Features

- **Declarative YAML pipelines** — write a recipe, run with `vkit run`
- **22 built-in operators** — segmentation, ASR, diarization, quality filtering, packaging
- **Resumable execution** — checkpoint every stage, resume after crashes
- **Disk-aware GC** — aggressive cleanup of intermediate derived audio
- **Inspectable results** — Rich CLI, self-contained HTML report, Gradio panel
- **Extensible** — plugin system for third-party operators and recipes

## Quick start

```bash
pip install voxkitchen
vkit init my-project
cd my-project
# Edit pipeline.yaml to point at your audio data
vkit run pipeline.yaml
vkit inspect run work/
```

## License

Apache 2.0. See [LICENSE](https://github.com/voxkitchen/voxkitchen/blob/main/LICENSE).
