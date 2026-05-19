# Examples

The docs page [Examples & Use Cases](../docs/examples.md) is the canonical
guide for choosing a pipeline by task. This directory contains the runnable
YAML files and small demo audio used by that guide.

## Demo data

`demo_data/` contains two opus files (~100 min total) for testing.

## Common pipelines

| File | What it does | Suggested image |
|------|-------------|-----------------|
| `minimal.yaml` | Identity passthrough to test the runner | `slim` |
| `demo-no-asr.yaml` | Small CPU-friendly demo with bundled audio | `slim` |
| `dir-resample-pack.yaml` | Directory ingest, resample, normalize, Kaldi export | `slim` |
| `data-cleaning.yaml` | Quality metrics, dedup, filtering, JSONL export | `slim` |
| `asr-training-data.yaml` | VAD, augmentation, ASR labeling, HuggingFace export | `asr` |
| `librispeech-asr.yaml` | LibriSpeech recipe ingest, ASR, quality filtering | `asr` |
| `speaker-analysis.yaml` | VAD, diarization, speaker/language annotations | `latest` |
| `tts-data-prep.yaml` | Clean, segment, transcribe, align, and pack TTS data | `asr` |

## Usage

```bash
# Validate a pipeline inside the prebuilt image without running it
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml --dry-run

# Run the demo pipeline (uses examples/demo_data/)
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml

# Inspect results
vkit inspect run work/demo-no-asr
```
