# TTS Tools Design Spec

## Goal

Add 3 TTS-oriented operators to VoxKitchen: speaker similarity scoring, neural audio codec tokenization, and CER/WER text accuracy metrics.

## Operators

### 1. `speaker_similarity` (quality)

Compute cosine similarity between each cut's speaker embedding and a reference embedding. Requires `speaker_embed` to have run first.

**Config:**
```python
class SpeakerSimilarityConfig(OperatorConfig):
    reference_path: str           # path to .npy file with reference embedding
    embedding_key: str = "speaker_embedding"  # key in cut.custom
```

**Behavior:**
- Load reference embedding from `.npy` file (saved via `np.save`)
- For each cut, read `cut.custom["speaker_embedding"]`, compute cosine similarity
- Write `metrics["speaker_similarity"]` (float, 0-1)
- Cuts without embeddings: write `metrics["speaker_similarity"] = 0.0` and log warning

**Dependencies:** numpy only (core dep). No new pip packages.

**Device:** cpu

**Usage example:**
```yaml
stages:
  - name: embed
    op: speaker_embed
    args: { method: wespeaker }
  - name: sim
    op: speaker_similarity
    args:
      reference_path: ./reference_speaker.npy
  - name: filter
    op: quality_score_filter
    args:
      conditions: ["metrics.speaker_similarity > 0.6"]
```

**tools.py convenience function:**
```python
def compute_speaker_similarity(audio_path, reference_path) -> float:
```

---

### 2. `codec_tokenize` (annotate)

Encode audio into discrete token sequences using a neural audio codec. Supports EnCodec and DAC backends.

**Config:**
```python
class CodecTokenizeConfig(OperatorConfig):
    backend: str = "encodec"       # "encodec" or "dac"
    bandwidth: float = 6.0         # target bandwidth in kbps (encodec)
    model: str = "encodec_24khz"   # model variant
```

**Behavior:**
- Load codec model in `setup()`
- For each cut: load audio → resample to codec's native SR (24kHz for EnCodec) → encode → extract token indices
- Write `custom["codec_tokens"]`: `list[list[int]]` — outer list = codebook layers, inner list = time steps
- Write `custom["codec_backend"]`, `custom["codec_sr"]`, `custom["codec_n_codebooks"]`

**Dependencies:**
- EnCodec: `pip install encodec` → new extras group `codec = ["encodec>=0.1"]`
- DAC: `pip install descript-audio-codec` → add to `codec` extras

**Device:** gpu (benefits from GPU but works on CPU)

**Usage example:**
```yaml
stages:
  - name: resample
    op: resample
    args: { target_sr: 24000 }
  - name: tokenize
    op: codec_tokenize
    args:
      backend: encodec
      bandwidth: 6.0
  - name: pack
    op: pack_jsonl
```

**tools.py convenience function:**
```python
def tokenize_audio(audio_path, backend="encodec") -> list[list[int]]:
```

---

### 3. `cer_wer` (quality)

Compute Character Error Rate (CER) and Word Error Rate (WER) between two text fields. Compares ASR-generated text against a ground truth text already present in the cut.

**Config:**
```python
class CerWerConfig(OperatorConfig):
    hypothesis_field: str = "text"          # supervision field with ASR output
    reference_field: str = "reference_text" # custom field with ground truth
```

**Behavior:**
- For each cut, extract hypothesis from `supervisions[0].text` and reference from `custom["reference_text"]`
- Compute CER (character-level edit distance / reference length)
- Compute WER (word-level edit distance / reference word count)
- Write `metrics["cer"]` and `metrics["wer"]` (float, 0-1)
- If reference is missing: skip, pass through unchanged

**Dependencies:** No new packages. Edit distance computed with stdlib `difflib` or a simple Levenshtein implementation.

**Device:** cpu

**Usage example:**
```yaml
# Assumes cuts have custom.reference_text from dataset recipe
stages:
  - name: asr
    op: faster_whisper_asr
    args: { model: large-v3 }
  - name: accuracy
    op: cer_wer
    args:
      hypothesis_field: text
      reference_field: reference_text
  - name: filter
    op: quality_score_filter
    args:
      conditions: ["metrics.cer < 0.1"]
```

---

## File Structure

```
Created:
  src/voxkitchen/operators/quality/speaker_similarity.py
  src/voxkitchen/operators/annotate/codec_tokenize.py
  src/voxkitchen/operators/quality/cer_wer.py
  tests/unit/operators/quality/test_speaker_similarity.py
  tests/unit/operators/annotate/test_codec_tokenize.py
  tests/unit/operators/quality/test_cer_wer.py
  examples/pipelines/tts-speaker-filter.yaml
  examples/pipelines/codec-tokenize.yaml

Modified:
  src/voxkitchen/operators/__init__.py      — register new operators
  pyproject.toml                             — add codec extras
  README.md                                  — update operator count
  src/voxkitchen/tools.py                    — add convenience functions
```

## pyproject.toml changes

```toml
codec = ["encodec>=0.1", "descript-audio-codec>=1.0"]
```

Add `codec` to `all` extras group.

## Testing Strategy

- `speaker_similarity`: pure numpy, no optional deps. Full unit tests.
- `codec_tokenize`: skip if encodec/dac not installed. Registration test + slow integration test.
- `cer_wer`: pure Python, no optional deps. Full unit tests with known text pairs.
