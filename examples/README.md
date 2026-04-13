# Example Pipelines

These YAML files demonstrate common VoxKitchen workflows.

## Files

- **minimal.yaml** — Identity passthrough. Good for testing your installation.
- **librispeech-asr.yaml** — Full pipeline: LibriSpeech recipe → resample → VAD → ASR → quality filter → HuggingFace pack. Requires GPU for ASR.
- **dir-resample-pack.yaml** — Scan a local directory, resample to 16kHz, normalize loudness, pack as Kaldi format.

## Usage

```bash
vkit validate examples/pipelines/minimal.yaml
vkit run examples/pipelines/dir-resample-pack.yaml
```
