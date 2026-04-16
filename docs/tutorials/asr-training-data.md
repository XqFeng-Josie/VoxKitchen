# ASR Training Data

Prepare augmented ASR training data with automatic transcription.

## Quick Start

```bash
pip install voxkitchen[audio,asr]
vkit init my-asr-project --template asr
cd my-asr-project
# Put your audio files in ./data/
vkit run pipeline.yaml
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

Download MUSAN noise data: `vkit download musan --root ./data/musan`

### Why HuggingFace output?

`pack_huggingface` produces a dataset loadable with `datasets.load_from_disk()`, which integrates directly with HuggingFace training pipelines (Transformers, SpeechBrain, ESPnet).

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
    root: /data/librispeech
    subsets: [train-clean-100]
```

### Kaldi output format

Replace `pack_huggingface` with:

```yaml
  - name: pack
    op: pack_kaldi
```

Produces `wav.scp`, `text`, `utt2spk`, `spk2utt` files.
