<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Common Voice

Mozilla's crowd-sourced multilingual ASR corpus covering 100+ languages; size, quality, and demographics vary widely by language.


- **Task:** asr, multilingual
- **Languages:** multi
- **Domain:** crowdsourced read speech
- **License:** CC0 1.0
- **Homepage:** [https://commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets)

## Recommendation

Best choice when you need a permissively-licensed ASR corpus for a low-resource language — likely the only freely available option for many languages. English and a handful of major languages have hundreds of hours; smaller languages may have only a few hours. Download a specific version/language snapshot for reproducibility.


## Getting the data

Downloadable via VoxKitchen (`commonvoice`, source: HuggingFace, size: —):

```bash
vkit docker download --tag slim commonvoice --root ./data/commonvoice
```

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
