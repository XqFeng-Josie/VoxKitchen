# Recipes & Dataset Download

VoxKitchen recipes parse popular speech datasets into CutSets. Some recipes also support automatic download.

## Available Recipes

| Recipe | Task | Language | Download | Description |
|--------|------|----------|:--------:|-------------|
| `librispeech` | ASR | English | openslr | Read-aloud audiobooks (960h) |
| `libritts` | TTS | English | openslr | Multi-speaker English TTS, sentence-segmented and TTS-normalized derivative of LibriSpeech |
| `ljspeech` | TTS | English | keithito | Single-speaker English TTS (~24h, 13k utterances) — canonical TTS baseline |
| `tedlium3` | ASR | English | openslr | TED-LIUM 3 — English TED talks (~452h), utterance-aligned via STM |
| `aishell` | ASR | Chinese | openslr | Mandarin read speech (170h) |
| `aishell3` | TTS | Chinese | openslr | Multi-speaker Mandarin TTS (218 speakers, ~85h) |
| `cnceleb` | Speaker | Chinese | openslr | CN-Celeb 1 — Chinese speaker recognition (~130k utts, 1000 spk, 11 genres) |
| `commonvoice` | ASR (multi) | Multi | manual | Mozilla crowdsourced recordings |
| `fleurs` | ASR / langid | 102 languages | HuggingFace | Google's multilingual eval set (~12h/lang) |
| `musan` | Augmentation | — | openslr | MUSAN — noise / music / speech augmentation source (~11 GB) |

## Downloading Datasets

```bash
# LibriSpeech (English, from openslr.org)
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets dev-clean
vkit docker download --tag slim librispeech --root ./data/librispeech --subsets train-clean-100

# LibriTTS (English multi-speaker TTS, from openslr.org)
vkit docker download --tag slim libritts --root ./data/libritts --subsets dev-clean
vkit docker download --tag slim libritts --root ./data/libritts --subsets train-clean-100

# LJSpeech (English single-speaker TTS, from data.keithito.com)
vkit docker download --tag slim ljspeech --root ./data/ljspeech

# TED-LIUM 3 (English TED-talk ASR, from openslr.org)
vkit docker download --tag slim tedlium3 --root ./data/tedlium3

# AISHELL-1 (Chinese ASR, from openslr.org)
vkit docker download --tag slim aishell --root ./data/aishell

# AISHELL-3 (Chinese multi-speaker TTS, from openslr.org)
vkit docker download --tag slim aishell3 --root ./data/aishell3

# CN-Celeb 1 (Chinese speaker recognition, from openslr.org)
vkit docker download --tag slim cnceleb --root ./data/cnceleb

# MUSAN (augmentation source — noise / music / speech, from openslr.org)
vkit docker download --tag slim musan --root ./data/musan

# FLEURS (multilingual, from HuggingFace)
vkit docker download --tag slim fleurs --root ./data/fleurs --subsets en_us,zh_cn,fr_fr
```

### LibriSpeech Subsets

| Subset | Hours | Description |
|--------|:-----:|-------------|
| `dev-clean` | 5.4 | Clean development set |
| `dev-other` | 5.3 | Noisy development set |
| `test-clean` | 5.4 | Clean test set |
| `test-other` | 5.1 | Noisy test set |
| `train-clean-100` | 100 | Clean training (recommended start) |
| `train-clean-360` | 363 | Clean training (large) |
| `train-other-500` | 496 | Noisy training |

### LibriTTS Subsets

Same partitioning as LibriSpeech (the corpus is a TTS-friendly
resegmentation of the same audio). Pick subsets the same way; the
recipe accepts the same names.

### LJSpeech

LJSpeech ships as a single archive (no subset selection). The recipe
emits one Cut per row of `metadata.csv` — 13,100 cuts, ~24 hours.
Each cut's supervision carries the *normalized* text; the *raw* text
is kept under `cut.custom["raw_text"]` only when normalization actually
changed it.

### AISHELL-3 Subsets

| Subset | Description |
|--------|-------------|
| `train` | 218 speakers, the bulk of the corpus (~85h) |
| `test`  | held-out evaluation split |

If a subset is missing on disk (partial extraction is common for the
~17 GB tarball) the recipe silently skips it.

### TED-LIUM 3 Subsets

| Subset | Hours | Description |
|--------|:-----:|-------------|
| `train` | ~268 | TED talks training split (most of the 452h corpus) |
| `dev`   | ~1.6 | Held-out dev set used by every TED-LIUM ASR paper |
| `test`  | ~2.6 | Held-out test set for reporting numbers |

Audio is NIST SPHERE (`.sph`) — libsndfile reads it natively, so no
manual conversion is needed. Each STM row produces one Cut that
references a `(start, duration)` slice of the parent talk's `.sph`.
Padding rows (`inter_segment_gap` speaker / `ignore_time_segment_in_scoring`
transcript) are dropped automatically.

### CN-Celeb 1 Subsets

| Subset | Description |
|--------|-------------|
| `data` | All FLAC under `data/` — every utterance from every speaker (~130k utts) |
| `dev`  | Utterances listed in `dev/dev.lst` — for tuning / model selection |
| `eval` | Concatenation of `eval/lists/enroll.lst` + `eval/lists/test.lst` — for trial pairs |

Default is `["data"]` when no subsets are passed. Asking for
`["data", "dev"]` is safe — overlapping utterances are deduplicated
by the recipe.

### MUSAN Subsets

| Subset | Description |
|--------|-------------|
| `noise` | Free-sound + sound-bible noise samples (recommended for SNR augmentation) |
| `music` | FMA + classical + jamendo musical clips |
| `speech` | Librivox + US-Gov English background speech (use for babble augmentation) |

Default is all three. MUSAN cuts carry no supervisions text — only the
audio plus `cut.custom["musan_category"]` / `["musan_subcategory"]`
tags. The natural consumer is the `noise_augment` operator, which
samples from any subset of these for mixing into your training data.

### CommonVoice (Manual Download)

CommonVoice requires a Mozilla account. Download from [commonvoice.mozilla.org](https://commonvoice.mozilla.org), then use the recipe:

```yaml
ingest:
  source: recipe
  recipe: commonvoice
  args:
    root: /path/to/cv-corpus-xx/en
    subsets: [train, test]
```

## Using Recipes in Pipelines

```yaml
version: "0.1"
name: my-pipeline
work_dir: ./work/${name}

ingest:
  source: recipe
  recipe: librispeech
  args:
    root: ./data/librispeech
    subsets: [train-clean-100]

stages:
  - name: resample
    op: resample
    args: { target_sr: 16000 }
  - name: pack
    op: pack_jsonl
```

## Adding Custom Recipes

See the [Contributing Guide](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/CONTRIBUTING.md#adding-a-new-recipe) for how to add new dataset recipes.
