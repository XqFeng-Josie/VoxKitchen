# VoxKitchen

Declarative speech data processing toolkit. Write a YAML recipe, run `vkit run`, get training-ready data.

> **Status:** Pre-alpha. API is unstable.

## Install

```bash
pip install voxkitchen

# For GPU operators (ASR, VAD):
pip install voxkitchen[asr]       # faster-whisper
pip install voxkitchen[segment]   # webrtcvad, librosa
pip install voxkitchen[viz-panel] # Gradio interactive panel
```

## 30-second quickstart

```bash
# Scaffold a project
vkit init my-project && cd my-project

# Run the pipeline
vkit run pipeline.yaml

# Inspect results
vkit inspect run work/
vkit inspect cuts work/01_pack/cuts.jsonl.gz
open work/report.html
```

## How it works

A pipeline is a YAML file: **ingest** raw audio, pass it through **stages** (operators), each stage transforms the data:

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${run_id}
ingest:
  source: dir
  args:
    root: /data/raw_audio
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000 }
  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }
  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3, compute_type: float16 }
  - name: filter
    op: quality_score_filter
    args:
      conditions: ["duration > 1", "duration < 30", "metrics.snr > 10"]
  - name: pack
    op: pack_huggingface
    args: { output_dir: ./output/hf_dataset }
```

## Operators

27 built-in operators across 5 categories:

| Category | Operators |
|----------|-----------|
| **Audio** | `resample`, `ffmpeg_convert`, `channel_merge`, `loudness_normalize` |
| **Segmentation** | `silero_vad`, `webrtc_vad`, `fixed_segment`, `silence_split` |
| **Annotation** | `faster_whisper_asr`, `whisperx_asr`, `paraformer_asr`, `sensevoice_asr`, `wenet_asr`, `pyannote_diarize`, `speechbrain_langid`, `gender_classify` |
| **Quality** | `snr_estimate`, `duration_filter`, `audio_fingerprint_dedup`, `quality_score_filter` |
| **Pack** | `pack_manifest`, `pack_huggingface`, `pack_webdataset`, `pack_parquet`, `pack_kaldi` |

```bash
# List all operators
vkit operators

# Show config fields + YAML example for any operator
vkit operators show silero_vad
```

## CLI commands

```
vkit init <path>              Scaffold a new project
vkit run <yaml>               Execute a pipeline
vkit run <yaml> --dry-run     Validate without executing
vkit validate <yaml>          Check YAML syntax
vkit ingest --source dir ...  Build CutSet without a pipeline
vkit inspect cuts <path>      CutSet statistics
vkit inspect run <work_dir>   Pipeline stage summary
vkit inspect trace <id> --in <work_dir>   Trace a cut's provenance
vkit inspect errors <work_dir>            Show per-cut errors
vkit operators                List all operators
vkit operators show <name>    Operator detail + config
vkit viz <path>               Gradio interactive explorer
```

## Python tools API

For quick tasks without writing a YAML pipeline:

```python
from voxkitchen.tools import transcribe, detect_speech, estimate_snr, audio_info

audio_info("speech.wav")
# AudioInfo(sample_rate=16000, duration=3.2, num_channels=1, format='WAV')

transcribe("speech.wav", model="tiny")
# [Segment(start=0.0, end=3.2, text="Hello world")]

detect_speech("speech.wav", method="silero")
# [(0.5, 2.8)]

estimate_snr("speech.wav")
# 18.3
```

## Key features

- **Resumable** — each stage checkpoints with `_SUCCESS` marker; crashes resume from last good stage
- **Error tolerant** — bad cuts are logged to `_errors.jsonl` and skipped, pipeline continues
- **GC** — intermediate derived audio is cleaned up automatically (`--keep-intermediates` to disable)
- **Provenance** — every Cut tracks which operator produced it from which parent
- **Extensible** — register custom operators via Python entry_points

## Examples

See [`examples/pipelines/`](examples/pipelines/) for ready-to-run YAML files.

## License

Apache 2.0. See [LICENSE](LICENSE).
