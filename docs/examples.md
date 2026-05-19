# Examples & Use Cases

Use this page when you already know what kind of speech data task you want to
run. Start with a template for normal projects; use the bundled example
pipelines for quick checks, demos, and advanced operator combinations.

## Quick Demo

The published Docker images include demo pipelines and demo audio, so you can
try VoxKitchen without cloning the repository:

```bash
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml --dry-run
vkit docker run --tag slim examples/pipelines/demo-no-asr.yaml
vkit inspect run ./work/demo-no-asr
```

Use this path to check that Docker, mounts, checkpoints, reports, and inspect
commands work on your machine.

## Start From A Template

Templates are the recommended starting point for real projects because they
create a local project directory with `data/`, `pipeline.yaml`, and a short
README.

| Goal | Command | Runtime image |
|---|---|---|
| Clean and filter raw speech audio | `vkit init my-cleaning --template cleaning` | `slim` |
| Build ASR training data | `vkit init my-asr --template asr` | `asr` |
| Analyze speakers and languages | `vkit init my-speakers --template speaker` | `latest` |
| Prepare TTS training data | `vkit init my-tts --template tts` | `asr` |

Typical run:

```bash
cd my-asr
cp /path/to/audio/* data/
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
vkit inspect run work/
```

## Bundled Example Pipelines

These YAML files are available inside the published Docker images under
`examples/pipelines/`. Clone the repository only if you want to inspect or edit
the files locally.

| Pipeline | Use case | Runtime image |
|---|---|---|
| `minimal.yaml` | Identity passthrough to check the runner | `slim` |
| `demo-no-asr.yaml` | Small CPU-friendly demo with bundled audio | `slim` |
| `demo-full.yaml` | Full demo with VAD, quality metrics, ASR, gender, filtering | `asr` |
| `dir-resample-pack.yaml` | Directory ingest, resample, normalize, Kaldi export | `slim` |
| `data-cleaning.yaml` | Quality metrics, dedup, filtering, JSONL export | `slim` |
| `asr-training-data.yaml` | VAD, augmentation, ASR labeling, HuggingFace export | `asr` |
| `librispeech-asr.yaml` | Recipe ingest from LibriSpeech, ASR, quality filter | `asr` |
| `qwen3-asr.yaml` | Qwen3-ASR transcription path | `asr` |
| `forced-align.yaml` | Word alignment for existing text/audio | `asr` |
| `speaker-analysis.yaml` | VAD, diarization, speaker/language annotations | `latest` |
| `speaker-embed.yaml` | Extract speaker embeddings from speech segments | `slim` |
| `tts-data-prep.yaml` | Clean, segment, transcribe, align, and pack TTS data | `asr` |
| `tts-synthesis.yaml` | Run a TTS synthesis operator | `tts` |
| `tts-speaker-filter.yaml` | Filter or inspect data by speaker metadata | `slim` |
| `speech-enhance.yaml` | Speech enhancement / denoising | `slim` |
| `augmentation.yaml` | Basic data augmentation flow | `slim` |
| `noise-augment.yaml` | Add noise augmentation from a noise directory | `slim` |
| `reverb-augment.yaml` | Add reverberation augmentation | `slim` |
| `emotion-recognize.yaml` | Emotion annotation | `asr` |
| `codec-tokenize.yaml` | Audio codec token extraction | `slim` |
| `fleurs-multilingual.yaml` | Multilingual recipe-style processing | `asr` |

Validate before running when you are unsure about paths, arguments, or image
choice:

```bash
vkit docker run --tag asr examples/pipelines/asr-training-data.yaml --dry-run
```

To edit these examples or write your own pipeline from scratch, see the
[Pipeline YAML reference](reference/pipeline-yaml.md).

## Choosing The Right Image

Use the smallest image that contains the operators in your pipeline:

| Image | Choose it for |
|---|---|
| `slim` | CPU-friendly prep, VAD, enhancement, quality checks, codec tokenization, packing |
| `asr` | ASR, forced alignment, and emotion annotation |
| `diarize` | Pyannote diarization only |
| `tts` | TTS engines except Fish-Speech |
| `fish-speech` | Fish-Speech isolated runtime |
| `latest` | Mixed pipelines across ASR, diarization, TTS, and Fish-Speech |

`vkit validate pipeline.yaml` prints the recommended pull/run command for your
pipeline.

## Inspect Outputs

Most examples write checkpoints and reports under `./work`; exporters usually
write final datasets under `./output`.

```bash
vkit inspect run work/
vkit inspect cuts work/<run>/<stage>/cuts.jsonl.gz
vkit inspect errors work/
```

For `pack_huggingface`, the audio column is embedded in the Arrow dataset. If
your training code needs decoded arrays, install `torchcodec`; for metadata or
custom decoding, read with `datasets.Audio(decode=False)`.
