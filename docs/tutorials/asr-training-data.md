# ASR Training Data

Prepare augmented ASR training data with automatic transcription.

## Quick Start

```bash
vkit init my-asr-project --template asr
cd my-asr-project
# Put your audio files in ./data/
vkit docker run --tag asr pipeline.yaml --dry-run
vkit docker run --tag asr pipeline.yaml
```

## What the Pipeline Does

| Stage | Operator | Why |
|-------|----------|-----|
| Resample | `resample` → 16kHz mono | ASR standard; most pretrained models expect 16kHz |
| VAD | `silero_vad` | Split long recordings into utterance-level segments |
| Speed perturb | `speed_perturb` [0.9, 1.0, 1.1] | 3x data augmentation — improves ASR robustness |
| Volume perturb | `volume_perturb` [-3, +3] dB | Simulates varying recording conditions |
| ASR | `faster_whisper_asr` large-v3 | Generate text labels for training |
| Filter | `quality_score_filter` | Remove too-short/too-long segments |
| Pack | `pack_huggingface` | Output as HuggingFace Dataset (ready for training) |

## Key Design Decisions

### Why speed perturbation?

Speed perturbation at factors [0.9, 1.0, 1.1] is the single most effective data augmentation for ASR. It simulates different speaking rates and slightly shifts pitch, making the model robust to natural variation. This triples your training data.

### Why not noise augmentation?

Noise augmentation (`noise_augment`) is also effective but requires a noise dataset (e.g., MUSAN). The default template keeps it simple — no external data dependencies. To add noise augmentation:

```yaml
  # Add after volume_aug, requires noise files in ./data/noise/
  - name: noise_aug
    op: noise_augment
    args:
      noise_dir: ./data/noise
      snr_range: [5, 20]
```

Download MUSAN with `vkit docker download --tag slim musan --root ./data/noise/musan`,
then point `noise_dir` at it — or place any other noise dataset under
`./data/noise/` before enabling this stage.

### Why HuggingFace output?

`pack_huggingface` produces a dataset loadable with `datasets.load_from_disk()`, which integrates directly with HuggingFace training pipelines (Transformers, SpeechBrain, ESPnet).

By default, the template writes the final dataset to `./output/hf_dataset`.
The audio column is embedded in the HuggingFace Dataset's Arrow shard, so the
final result is not a directory of standalone WAV files. Use
`--keep-intermediates` or `gc_mode: keep` when you also need to preserve each
stage's derived WAV files under `./work`.

Load the exported dataset with:

```python
from datasets import load_from_disk

ds = load_from_disk("./output/hf_dataset")
```

Recent HuggingFace `datasets` versions decode `Audio` columns through
`torchcodec`. If your training code needs `audio["array"]`, install it in the
training environment:

```bash
pip install torchcodec
```

For metadata checks or custom audio decoding, avoid automatic decode:

```python
from datasets import Audio, load_from_disk

ds = load_from_disk("./output/hf_dataset")
ds = ds.cast_column("audio", Audio(decode=False))
row = ds[0]
audio_bytes = row["audio"]["bytes"]
```

## Customization

### For Chinese ASR

Replace `faster_whisper_asr` with Qwen3 or Paraformer:

```yaml
  - name: asr
    op: qwen3_asr
    args:
      model: Qwen/Qwen3-ASR-0.6B
      language: Chinese

  # Or use Paraformer (optimized for Chinese)
  - name: asr
    op: paraformer_asr
    args:
      model: iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

### Start from a known dataset

Replace `dir` ingest with a recipe:

```yaml
ingest:
  source: recipe
  recipe: librispeech
  args:
    root: ./data/librispeech
    subsets: [train-clean-100]
```

### Kaldi output format

Replace `pack_huggingface` with:

```yaml
  - name: pack
    op: pack_kaldi
```

Produces `wav.scp`, `text`, `utt2spk`, `spk2utt` files.
