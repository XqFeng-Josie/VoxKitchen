# TTS Synthesis

Generate speech from text using VoxKitchen's four TTS engines. Pick the
engine based on what you need (built-in voices, voice cloning, language
coverage) and the runtime budget (CPU vs GPU, image size).

For preparing audio data to train your own TTS model, see the companion
tutorial: [TTS Data Preparation](tts-data-prep.md).

## Quick Start (Python)

The fastest path is `voxkitchen.tools.synthesize` from inside a Docker
shell — one line per utterance, no YAML.

```python
from voxkitchen.tools import synthesize

# Built-in English voice, runs on CPU
synthesize("Hello world!", "out.wav", engine="kokoro")

# Built-in Mandarin voice
synthesize("你好，世界。", "zh.wav", engine="kokoro", language="z")
```

Run from inside an image that has the right engine installed:

```bash
vkit docker shell --tag tts
# inside the container:
python -c "from voxkitchen.tools import synthesize; synthesize('Hello!', '/app/output/hello.wav', engine='kokoro')"
```

## Quick Start (YAML)

Batch synthesis runs through the standard pipeline: ingest a manifest of
text-only cuts, then a `tts_<engine>` stage, then pack the results.

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/docs/schemas/pipeline.schema.json
version: "0.1"
name: tts-synthesis
work_dir: ./work/${name}

ingest:
  source: manifest
  args:
    path: ./text_scripts.jsonl.gz

stages:
  - name: synthesize
    op: tts_kokoro
    args:
      voice: af_heart
      lang_code: a
      speed: 1.0

  - name: pack
    op: pack_jsonl
```

The input manifest is a regular VoxKitchen `cuts.jsonl.gz` whose
supervisions carry `text` (no audio needed). Each text supervision is
synthesized into one new WAV file written under
`<work_dir>/<stage>/derived/<cut-id>__<engine>.wav`, and the output
manifest references those files.

A ready-to-use example lives at
[`examples/pipelines/tts-synthesis.yaml`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/examples/pipelines/tts-synthesis.yaml).

## Engine Capabilities

| Engine | Cloning | Built-in voices | Reference audio | Language scope | Device | Output SR | Image |
|---|:---:|:---:|:---:|---|:---:|:---:|:---:|
| `tts_kokoro` | — | yes (e.g. `af_heart`) | — | 8 codes: `a` AmE, `b` BrE, `j` JA, `z` ZH, `f` FR, `h` HI, `i` IT, `p` BR-PT | CPU | 24 kHz | `tts` |
| `tts_chattts` | — | seed-sampled (set `seed` to reproduce) | — | Chinese & English | GPU | 24 kHz | `tts` |
| `tts_cosyvoice` | ✓ | yes (`sft` mode, `spk_id`) | for `zero_shot` / `cross_lingual` | Multilingual (zh, en, ja, ko, …) | GPU | 24 kHz | `tts` |
| `tts_fish_speech` | ✓ | — (reference-driven) | required | Language-agnostic, follows reference accent | GPU | 44.1 kHz | `fish-speech` |

The first three engines share the `tts` image. Fish-Speech lives in its
own `fish-speech` image because upstream pins `torch==2.8`, incompatible
with the other three engines' stack. Mixing engines or mixing TTS with
ASR/diarize stages requires the `latest` image.

## Use Case 1 — Built-in Voice

For neutral system voices, simple read-aloud, or quick demos.

### Kokoro (CPU, 8 languages)

```yaml
- name: synthesize
  op: tts_kokoro
  args:
    voice: af_heart       # English female; see Kokoro voice list
    lang_code: a          # a=AmE, b=BrE, j=Japanese, z=Mandarin, …
    speed: 1.0            # 0.8 = slower, 1.2 = faster
```

Kokoro is the only engine that runs comfortably on CPU and the only one
shipping a wide language menu. Use it as a default when GPU is scarce.

### CosyVoice (GPU, `sft` mode, built-in speakers)

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: sft
    spk_id: default       # or another built-in speaker id
```

`sft` skips the reference-audio path entirely, so it's the fastest of
CosyVoice's three modes.

### ChatTTS (GPU, conversational style)

```yaml
- name: synthesize
  op: tts_chattts
  args:
    seed: 42              # fix to keep speaker timbre stable across calls
    temperature: 0.3
```

`seed` controls the sampled timbre. Leave it `null` for a different
"voice" each call. Prosody tags like `[laugh]`, `[uv_break]` can be
embedded directly in the text.

## Use Case 2 — Voice Cloning From A Short Reference

Both `tts_cosyvoice` and `tts_fish_speech` clone a voice from a single
short reference recording (3–10 seconds is enough for both).

### CosyVoice `zero_shot` — reference audio + transcript

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: zero_shot
    reference_audio: ./voices/alice_5s.wav   # 3–10 second WAV
    reference_text: "我是参考文本，时长大约五秒。"
```

The transcript anchors timbre extraction; the model then synthesizes any
new text in the same voice and language.

### CosyVoice `cross_lingual` — voice across languages

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: cross_lingual
    reference_audio: ./voices/alice_en_5s.wav
    # No reference_text needed; speak any supported language.
```

Use this when the reference is in one language but the output should be
in another (e.g. English reference, Mandarin output).

### Fish-Speech — reference-driven, no built-in voices

```yaml
- name: synthesize
  op: tts_fish_speech
  args:
    reference_audio: ./voices/alice_5s.wav
    reference_text: "I am the reference, about five seconds long."
    seed: 42                # optional; locks token sampling
    temperature: 0.8
```

Fish-Speech outputs 44.1 kHz audio. It ships in its own `fish-speech`
image (~57 GB). Use the `latest` image only when mixing Fish-Speech with
other engines in one pipeline.

### Python (any cloning engine)

```python
from voxkitchen.tools import synthesize

# CosyVoice zero-shot clone
synthesize(
    "你好，这是克隆出来的声音。",
    "out.wav",
    engine="cosyvoice",
    reference_audio="ref.wav",
    reference_text="我是参考文本。",
)

# Fish-Speech clone
synthesize(
    "Hello, this is a cloned voice.",
    "out.wav",
    engine="fish_speech",
    reference_audio="ref.wav",
)
```

## Choosing An Engine

A short decision flow:

- **Need to run without a GPU?** → `tts_kokoro`.
- **Need to clone a specific voice from a short clip?** → `tts_cosyvoice`
  (zero_shot or cross_lingual) for 24 kHz, `tts_fish_speech` for 44.1
  kHz.
- **Mandarin or English read-aloud with a "natural" feel?** →
  `tts_chattts` (conversational); pin `seed` for stability.
- **Multilingual coverage matters more than expressiveness?** →
  `tts_kokoro` (8 languages including ZH/JA).

When in doubt, run

```bash
vkit validate pipeline.yaml
```

— it prints the smallest Docker tag that can run all stages in your
pipeline.

## Preparing The Input Manifest

The synthesis pipeline reads a `cuts.jsonl.gz` whose cuts already carry
text. Two common ways to produce one:

1. **Write Python** that emits `Cut` records with `Supervision(text=...)`
   and `to_jsonl_gz` to disk. See
   [Python Tools API](../reference/tools-api.md) for `Cut` / `CutSet`.
2. **Reuse the output of an ASR pipeline** — every cut from an ASR run
   already has `text`. Point this synthesis pipeline at that manifest to
   resynthesize the same utterances with a different voice.

## Inspecting Outputs

```bash
vkit inspect run work/tts-synthesis/
vkit inspect cuts work/tts-synthesis/01_pack/cuts.jsonl.gz
```

The synthesized WAV files live under each TTS stage's `derived/`
directory. The final manifest's recordings point at them.

## Related

- [TTS Data Preparation](tts-data-prep.md) — produce clean speech for
  training a TTS model.
- [Python Tools API — TTS Synthesis](../reference/tools-api.md#tts-synthesis)
  — full signature of `synthesize()`.
- [Operators reference](../reference/operators.md) — every TTS operator's
  full config schema.
