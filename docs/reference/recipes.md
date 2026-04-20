# Recipes & Dataset Download

VoxKitchen recipes parse popular speech datasets into CutSets. Some recipes also support automatic download.

## Available Recipes

| Recipe | Language | Download | Description |
|--------|----------|:--------:|-------------|
| `librispeech` | English | openslr | Read-aloud audiobooks (960h) |
| `aishell` | Chinese | openslr | Mandarin read speech (170h) |
| `commonvoice` | Multi | manual | Mozilla crowdsourced recordings |
| `fleurs` | 102 languages | HuggingFace | Google's multilingual eval set (~12h/lang) |

## Downloading Datasets

```bash
# LibriSpeech (English, from openslr.org)
vkit download librispeech --root /data/librispeech --subsets dev-clean
vkit download librispeech --root /data/librispeech --subsets train-clean-100

# AISHELL-1 (Chinese, from openslr.org)
vkit download aishell --root /data/aishell

# FLEURS (multilingual, from HuggingFace)
vkit download fleurs --root /data/fleurs --subsets en_us,zh_cn,fr_fr
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
    root: /data/librispeech
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
