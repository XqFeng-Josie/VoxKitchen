# Getting Started

## Installation

```bash
# Create virtual environment
conda create -n voxkitchen python=3.11 -y
conda activate voxkitchen

# Clone and install
git clone https://github.com/voxkitchen/voxkitchen.git
cd voxkitchen
pip install -e .

# Install extras for GPU server (ASR, quality analysis, etc.)
pip install -e ".[asr,whisper,pitch,dnsmos,segment]"
```

## Your first pipeline

### 1. Initialize a project

```bash
vkit init my-project
cd my-project
```

This creates a `pipeline.yaml` and a `data/` directory.

### 2. Add your audio

Put some `.wav`, `.flac`, or `.mp3` files into the `data/` directory.

### 3. Edit pipeline.yaml

Replace the contents with:

```yaml
version: "0.1"
name: my-first-pipeline
work_dir: ./work/${run_id}

ingest:
  source: dir
  args:
    root: ./data
    recursive: true

stages:
  - name: resample
    op: resample
    args:
      target_sr: 16000
      target_channels: 1

  - name: pack
    op: pack_manifest
```

This pipeline scans `data/`, resamples everything to 16kHz mono, and writes a manifest.

### 4. Validate (optional)

```bash
vkit run pipeline.yaml --dry-run
```

### 5. Run

```bash
vkit run pipeline.yaml
```

### 6. Inspect results

```bash
# Stage summary
vkit inspect run work/

# Cut statistics
vkit inspect cuts work/01_pack/cuts.jsonl.gz

# Open the auto-generated HTML report
open work/report.html
```

## Discover operators

```bash
# List all 27 operators
vkit operators

# Show config fields for any operator
vkit operators show silero_vad
vkit operators show faster_whisper_asr
```

## Standalone tools

For quick one-off tasks without writing a YAML pipeline:

```bash
# Ingest audio without a pipeline
vkit ingest --source dir --root /path/to/audio --out my-cuts.jsonl.gz

# Interactive exploration (requires pip install voxkitchen[viz-panel])
vkit viz my-cuts.jsonl.gz
```

Or use the Python API directly:

```python
from voxkitchen.tools import transcribe, estimate_snr, audio_info

audio_info("speech.wav")
transcribe("speech.wav", model="tiny")
estimate_snr("speech.wav")
```

## Next steps

- Browse [example pipelines](../examples/pipelines/) for real-world workflows
- Read [Data Protocol](concepts/data-protocol.md) to understand Recording / Supervision / Cut
- Run `vkit operators show <name>` for any operator's config reference
