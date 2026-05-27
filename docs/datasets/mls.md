<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Multilingual LibriSpeech

50,000-hour multilingual audiobook ASR corpus derived from LibriVox recordings covering 8 languages (English, German, Dutch, French, Spanish, Italian, Portuguese, Polish).

- **Task:** asr, multilingual
- **Languages:** multi
- **Hours:** 50000
- **Domain:** audiobook
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/94](https://www.openslr.org/94)
- **Paper:** [https://arxiv.org/abs/2012.03411](https://arxiv.org/abs/2012.03411)

## Recommendation

Best choice for large-scale multilingual ASR training with a permissive license. English component alone is 44,500 h. Non-English languages range from 500–2,400 h — enough for competitive baselines. Audiobook domain only; supplement with conversational data if needed.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/94).

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
