# Voice Cloning & TTS

Clone a target voice from a single short reference recording (3–10
seconds is enough) and synthesize new text in that voice. VoxKitchen
ships two reference-driven engines: **CosyVoice** (24 kHz, GPU, `tts`
image) and **Fish-Speech** (44.1 kHz, GPU, `fish-speech` image).

> Just want a built-in voice and don't need cloning? See
> [Speaker TTS](tts-speaker.md). Preparing audio to *train* your own
> TTS model? See [TTS Training Data](tts-training-data.md).

## Quick Start (Python)

```python
from voxkitchen.tools import synthesize

# CosyVoice zero-shot — Mandarin reference + transcript
synthesize(
    "你好，这是克隆出来的声音。",
    "out.wav",
    engine="cosyvoice",
    reference_audio="ref.wav",
    reference_text="我是参考文本。",
)

# Fish-Speech — reference + transcript, English
synthesize(
    "Hello, this is a cloned voice.",
    "out.wav",
    engine="fish_speech",
    reference_audio="ref.wav",
    reference_text="I am the reference, about five seconds long.",
)
```

Run from inside an image that has the right engine installed:

```bash
vkit docker shell --tag tts          # CosyVoice
vkit docker shell --tag fish-speech  # Fish-Speech
```

## Quick Start (YAML)

Batch cloning runs through the standard pipeline: ingest a manifest of
text-only cuts, run a `tts_<engine>` stage that points at a reference
WAV, then pack the results.

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/docs/schemas/pipeline.schema.json
version: "0.1"
name: tts-voice-cloning
work_dir: ./work/${name}

ingest:
  source: manifest
  args:
    path: ./text_scripts.jsonl.gz

stages:
  - name: clone
    op: tts_cosyvoice
    args:
      mode: zero_shot
      reference_audio: ./voices/alice_5s.wav
      reference_text: "我是参考文本，时长大约五秒。"

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

| Engine / mode | Reference text | Cross-lingual | Device | Output SR | Image |
|---|:---:|:---:|:---:|:---:|:---:|
| `tts_cosyvoice` (`zero_shot`) | required | — (same language as reference) | GPU | 24 kHz | `tts` |
| `tts_cosyvoice` (`cross_lingual`) | not used | yes | GPU | 24 kHz | `tts` |
| `tts_fish_speech` | optional but recommended | follows reference accent | GPU | 44.1 kHz | `fish-speech` |

Fish-Speech lives in its own `fish-speech` image because upstream pins
`torch==2.8`, incompatible with the CosyVoice stack. Mixing the two
engines in one pipeline — or mixing either with ASR / diarization
stages — requires the `latest` image.

## Preparing The Reference Audio

A clean 3–10 second reference is enough for both engines. The clip
quality dominates the output far more than its length, so prefer:

- **Single speaker**, no overlap, no background music.
- **Mono**, with the engine's sample rate or higher: 24 kHz+ for
  CosyVoice, 44.1 kHz+ for Fish-Speech. The engines resample if needed.
- **Loudness in a normal speaking range** — clipping or whispered audio
  will be cloned along with the timbre.
- For `zero_shot` and Fish-Speech: an **accurate transcript** of the
  reference. The transcript anchors timbre extraction; small mistakes
  in this string materially degrade output.

If you only have a long recording, segment to a clean 3–10 s slice
first — `vkit init my-voices --template tts` scaffolds a pipeline that
denoises, segments, and transcribes raw audio into exactly this shape.
See [TTS Training Data](tts-training-data.md).

## CosyVoice — `zero_shot` mode

Reference audio plus its transcript. Output is synthesized in the
reference speaker's voice and language.

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: zero_shot
    reference_audio: ./voices/alice_5s.wav   # 3–10 second WAV
    reference_text: "我是参考文本，时长大约五秒。"
```

The transcript anchors timbre extraction; the model then synthesizes
any new text in the same voice and language.

## CosyVoice — `cross_lingual` mode

Voice carries over across languages — English reference, Mandarin
output (or vice versa). No reference transcript needed.

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: cross_lingual
    reference_audio: ./voices/alice_en_5s.wav
    # No reference_text needed; the input text can be in any supported language.
```

## Fish-Speech — 44.1 kHz, language-agnostic

```yaml
- name: synthesize
  op: tts_fish_speech
  args:
    reference_audio: ./voices/alice_5s.wav
    reference_text: "I am the reference, about five seconds long."
    seed: 42                # optional; locks token sampling
    temperature: 0.8
```

Fish-Speech outputs 44.1 kHz audio and follows the reference accent
across languages. It ships in its own `fish-speech` image (~57 GB);
use the `latest` image only when mixing Fish-Speech with other engines
in one pipeline.

## Choosing An Engine

- **Same-language cloning, 24 kHz acceptable** → `tts_cosyvoice` in
  `zero_shot` mode.
- **Voice from one language, synthesis in another** → `tts_cosyvoice`
  in `cross_lingual` mode.
- **Need 44.1 kHz output, or following the reference accent across
  languages** → `tts_fish_speech`.
- **No GPU available** → cloning is not currently supported on CPU; use
  [Speaker TTS](tts-speaker.md) with `tts_kokoro` instead.

When in doubt, run

```bash
vkit validate pipeline.yaml
```

— it prints the smallest Docker tag that can run all stages in your
pipeline.

## Preparing The Input Manifest

The synthesis pipeline reads a `cuts.jsonl.gz` whose cuts already
carry text. Two common ways to produce one:

1. **Write Python** that emits `Cut` records with `Supervision(text=...)`
   and `to_jsonl_gz` to disk. See
   [Python Tools API](../reference/tools-api.md) for `Cut` / `CutSet`.
2. **Reuse the output of an ASR pipeline** — every cut from an ASR run
   already has `text`. Point this cloning pipeline at that manifest to
   resynthesize the same utterances in the reference voice.

## Inspecting Outputs

```bash
vkit inspect run work/tts-voice-cloning/
vkit inspect cuts work/tts-voice-cloning/01_pack/cuts.jsonl.gz
```

The synthesized WAV files live under each TTS stage's `derived/`
directory. The final manifest's recordings point at them.

## Related

- [Speaker TTS](tts-speaker.md) — pick a built-in voice instead of
  cloning one.
- [TTS Training Data](tts-training-data.md) — quality gate for raw
  speech, including how to slice a long recording down to a clean
  reference clip.
- [Python Tools API — TTS Synthesis](../reference/tools-api.md#tts-synthesis)
  — full signature of `synthesize()`.
- [Operators reference](../reference/operators.md) — every TTS
  operator's full config schema.
