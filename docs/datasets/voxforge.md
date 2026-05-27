<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# VoxForge

Community-contributed crowdsourced corpus of transcribed read speech collected to build free, open acoustic models for open-source ASR engines.

- **Task:** asr, multilingual
- **Languages:** multi
- **Domain:** crowdsourced read
- **License:** GNU GPL
- **Homepage:** [https://www.voxforge.org/](https://www.voxforge.org/)

## Recommendation

Good for free, copyleft-licensed ASR acoustic-model training and as a multilingual baseline, especially where GPL licensing is acceptable. Crowdsourced quality and per-language volume vary widely, and the strong copyleft can complicate combining it with differently-licensed data.

## Getting the data

Obtain from the [dataset homepage](https://www.voxforge.org/).

English is the largest of several languages (also de, ru, es, fr, it, ...). GPL applies to submitted audio; exact GPL version not authoritatively stated.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
