# Recipes & Dataset Download

VoxKitchen recipes parse popular speech datasets into CutSets. Some recipes also support automatic download.

## Available Recipes

| Recipe | Task | Language | Download | Description |
|--------|------|----------|:--------:|-------------|
| `librispeech` | ASR | English | openslr | Read-aloud audiobooks (960h) |
| `libritts` | TTS | English | openslr | Multi-speaker English TTS, sentence-segmented and TTS-normalized derivative of LibriSpeech |
| `ljspeech` | TTS | English | keithito | Single-speaker English TTS (~24h, 13k utterances) — canonical TTS baseline |
| `aishell` | ASR | Chinese | openslr | Mandarin read speech (170h) |
| `aishell3` | TTS | Chinese | openslr | Multi-speaker Mandarin TTS (218 speakers, ~85h) |
| `commonvoice` | ASR (multi) | Multi | manual | Mozilla crowdsourced recordings |
| `fleurs` | ASR / langid | 102 languages | HuggingFace | Google's multilingual eval set (~12h/lang) |

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

# AISHELL-1 (Chinese ASR, from openslr.org)
vkit docker download --tag slim aishell --root ./data/aishell

# AISHELL-3 (Chinese multi-speaker TTS, from openslr.org)
vkit docker download --tag slim aishell3 --root ./data/aishell3

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
