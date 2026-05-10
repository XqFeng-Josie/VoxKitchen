# Examples

## Demo data

`demo_data/` contains two opus files (~100 min total) for testing.

## Pipelines

| File | What it does | Suggested image |
|------|-------------|-----------------|
| `minimal.yaml` | Identity passthrough — test your installation | `slim` |
| `dir-resample-pack.yaml` | Scan dir → resample 16kHz → normalize → Kaldi pack | `slim` |
| `librispeech-asr.yaml` | LibriSpeech → VAD → ASR → quality filter → HuggingFace | `asr` |
| `demo-full.yaml` | opus → wav → VAD → SNR → ASR → gender → filter → pack | `asr` |

## Usage

```bash
# Validate a pipeline inside the prebuilt image without running it
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml --dry-run

# Run the demo pipeline (uses examples/demo_data/)
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml

# Inspect results
vkit inspect run work/demo-no-asr
```
