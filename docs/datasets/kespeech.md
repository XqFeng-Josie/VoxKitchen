<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# KeSpeech

1,542 h from 27,237 speakers across 34 cities, covering standard Mandarin and its 8 subdialects with transcription, speaker, and subdialect labels.

- **Task:** asr, multilingual
- **Languages:** zh
- **Hours:** 1542
- **Domain:** dialect
- **License:** see source terms
- **Homepage:** [https://github.com/KeSpeech/KeSpeech](https://github.com/KeSpeech/KeSpeech)
- **Paper:** [https://openreview.net/forum?id=b3Zoeq2sCLq](https://openreview.net/forum?id=b3Zoeq2sCLq)

## Recommendation

Excellent for Mandarin ASR, accent/subdialect identification, and speaker recognition, with parallel Mandarin-vs-subdialect recordings. Choose it when dialectal robustness or large speaker diversity matters. Download requires accepting a custom usage agreement.

## Getting the data

Obtain from the [dataset homepage](https://github.com/KeSpeech/KeSpeech).

NeurIPS 2021 Datasets & Benchmarks. Download requires accepting a usage agreement; covers 8 subdialects.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
