# Getting Started

## Installation

```bash
pip install voxkitchen
```

For GPU-accelerated operators (ASR, VAD):
```bash
pip install voxkitchen torch torchaudio
pip install voxkitchen[asr]  # adds faster-whisper
```

## Your first pipeline

### 1. Initialize a project

```bash
vkit init my-project
cd my-project
```

This creates `pipeline.yaml` and `README.md`.

### 2. Prepare your data

Place your audio files in a `data/` directory, or edit `pipeline.yaml` to point at an existing directory.

### 3. Run the pipeline

```bash
vkit run pipeline.yaml
```

### 4. Inspect results

```bash
# Terminal summary
vkit inspect run work/

# View detailed statistics
vkit inspect cuts work/01_pack/cuts.jsonl.gz

# Open HTML report
open work/report.html
```

## Standalone ingest

You can also ingest audio without running a full pipeline:

```bash
vkit ingest --source dir --root /path/to/audio --out my-cuts.jsonl.gz
```

## Validation

Check a pipeline YAML for errors before running:

```bash
vkit validate pipeline.yaml
```
