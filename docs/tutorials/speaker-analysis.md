# Speaker Analysis

Analyze speaker distribution in audio data: identify speakers, extract embeddings, classify gender and language.

## Quick Start

```bash
pip install voxkitchen[audio,diarize,speaker,classify]
vkit init my-speaker-project --template speaker
cd my-speaker-project

# Set HuggingFace token (required for pyannote diarization)
echo "HF_TOKEN=hf_your_token" > .env

# Put your audio files in ./data/
vkit run pipeline.yaml
```

## What the Pipeline Does

| Stage | Operator | Output |
|-------|----------|--------|
| Resample | `resample` → 16kHz | Normalized audio |
| VAD | `silero_vad` | Speech segments |
| Diarize | `pyannote_diarize` | Speaker labels (spk_0, spk_1, ...) |
| Embed | `speaker_embed` (WeSpeaker) | 512-d speaker embedding vector per segment |
| Gender | `gender_classify` (F0-based) | "m" / "f" / "o" per segment |
| Language | `whisper_langid` | Detected language per segment |
| Pack | `pack_jsonl` | Manifest with all annotations |

## Use Cases

### Speaker counting

After running the pipeline, count unique speakers:

```python
from voxkitchen.schema.cutset import CutSet

cuts = CutSet.from_jsonl_gz("work/.../06_pack/cuts.jsonl.gz")
speakers = set()
for cut in cuts:
    for sup in cut.supervisions:
        if sup.speaker:
            speakers.add(sup.speaker)
print(f"Found {len(speakers)} speakers")
```

### Speaker similarity (using embeddings)

Compare two speakers using cosine similarity:

```python
import numpy as np

# Get embeddings from two cuts
emb1 = np.array(cut1.custom["speaker_embedding"])
emb2 = np.array(cut2.custom["speaker_embedding"])

# Cosine similarity (1.0 = same speaker, 0.0 = different)
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Similarity: {similarity:.3f}")
# > 0.65 → likely same speaker
# < 0.40 → likely different speakers
```

### Gender distribution

```python
from collections import Counter

genders = Counter()
for cut in cuts:
    for sup in cut.supervisions:
        if sup.gender:
            genders[sup.gender] += 1
print(genders)  # Counter({'m': 150, 'f': 120, 'o': 5})
```

## Customization

### Without diarization (simpler, no HF_TOKEN needed)

Remove the diarize stage if your audio is already single-speaker per file:

```yaml
stages:
  - name: resample
    op: resample
    args: { target_sr: 16000, target_channels: 1 }
  - name: vad
    op: silero_vad
    args: { threshold: 0.5 }
  - name: embed
    op: speaker_embed
    args: { method: wespeaker }
  - name: pack
    op: pack_jsonl
```

### Using SpeechBrain for speaker embeddings

```yaml
  - name: embed
    op: speaker_embed
    args:
      method: speechbrain
      speechbrain_model: speechbrain/spkrec-ecapa-voxceleb
```

### Add emotion recognition

```yaml
  - name: emotion
    op: emotion_recognize
    args:
      model: iic/emotion2vec_plus_large
```
