# Speaker TTS

Synthesize text in a **built-in voice** — a named voice that ships with
the engine, or a seed-sampled timbre you can reproduce by fixing the
seed. No reference audio required.

> Need to clone a specific voice from a short recording instead? See
> [Voice Cloning & TTS](tts-voice-cloning.md). Preparing audio to
> *train* your own TTS model? See [TTS Training Data](tts-training-data.md).

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

Batch synthesis runs through the standard pipeline: ingest a manifest
of text-only cuts, then a `tts_<engine>` stage, then pack the results.

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/XqFeng-Josie/VoxKitchen/main/docs/schemas/pipeline.schema.json
version: "0.1"
name: tts-speaker
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

| Engine | Voice source | Language scope | Device | Output SR | Image |
|---|---|---|:---:|:---:|:---:|
| `tts_kokoro` | named voices (e.g. `af_heart`) | 8 codes: `a` AmE, `b` BrE, `j` JA, `z` ZH, `f` FR, `h` HI, `i` IT, `p` BR-PT | CPU | 24 kHz | `tts` |
| `tts_chattts` | seed-sampled timbre (set `seed` to reproduce) | Chinese & English | GPU | 24 kHz | `tts` |
| `tts_cosyvoice` (`sft` mode) | built-in `spk_id` (no reference audio) | Multilingual (zh, en, ja, ko, …) | GPU | 24 kHz | `tts` |

All three engines share the `tts` image. Mixing them with ASR or
diarization stages in a single pipeline requires `latest`.

## Kokoro — CPU, 8 languages

```yaml
- name: synthesize
  op: tts_kokoro
  args:
    voice: af_heart       # English female; see Kokoro voice list
    lang_code: a          # a=AmE, b=BrE, j=Japanese, z=Mandarin, …
    speed: 1.0            # 0.8 = slower, 1.2 = faster
```

Kokoro is the only engine in this group that runs comfortably on CPU
and the only one shipping a wide language menu. Use it as a default
when GPU is scarce or when you need multilingual coverage.

## CosyVoice — `sft` mode, built-in speakers

```yaml
- name: synthesize
  op: tts_cosyvoice
  args:
    mode: sft
    spk_id: default       # or another built-in speaker id
```

`sft` skips the reference-audio path entirely, so it's the fastest of
CosyVoice's three modes. For the reference-driven modes (`zero_shot`,
`cross_lingual`), see [Voice Cloning & TTS](tts-voice-cloning.md).

## ChatTTS — conversational style, seed-driven

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

## Choosing An Engine

- **Need to run without a GPU?** → `tts_kokoro`.
- **Multilingual coverage matters most?** → `tts_kokoro` (8 languages
  including ZH / JA).
- **Mandarin or English read-aloud with a "natural" feel?** →
  `tts_chattts`; pin `seed` for stability.
- **Want a fixed named speaker on GPU?** → `tts_cosyvoice` in `sft`
  mode.

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
vkit inspect run work/tts-speaker/
vkit inspect cuts work/tts-speaker/01_pack/cuts.jsonl.gz
```

The synthesized WAV files live under each TTS stage's `derived/`
directory. The final manifest's recordings point at them.

## Related

- [Voice Cloning & TTS](tts-voice-cloning.md) — clone a voice from a
  short reference instead of picking a built-in one.
- [TTS Training Data](tts-training-data.md) — quality gate for raw
  speech you want to use to train your own TTS model.
- [Python Tools API — TTS Synthesis](../reference/tools-api.md#tts-synthesis)
  — full signature of `synthesize()`.
- [Operators reference](../reference/operators.md) — every TTS
  operator's full config schema.
