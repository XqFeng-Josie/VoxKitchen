# Examples

## Demo data

`demo_data/` contains two opus files (~100 min total) for testing.

## Pipelines

| File | What it does | Requirements |
|------|-------------|--------------|
| `minimal.yaml` | Identity passthrough — test your installation | None |
| `dir-resample-pack.yaml` | Scan dir → resample 16kHz → normalize → Kaldi pack | pyloudnorm |
| `librispeech-asr.yaml` | LibriSpeech → VAD → ASR → quality filter → HuggingFace | GPU, faster-whisper |
| `demo-full.yaml` | opus → wav → VAD → SNR → ASR → gender → filter → pack | torch, faster-whisper, librosa |

## Usage

```bash
# Validate a pipeline without running it
vkit run examples/pipelines/demo-full.yaml --dry-run

# Run the demo pipeline (uses examples/demo_data/)
vkit run examples/pipelines/demo-full.yaml

# Inspect results
vkit inspect run work/demo-full-*/
vkit inspect cuts work/demo-full-*/06_pack/cuts.jsonl.gz
```
