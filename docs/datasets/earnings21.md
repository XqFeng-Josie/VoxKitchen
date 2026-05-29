<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Earnings-21

39 hours of 44 English-language earnings calls from 2020 across nine financial sectors, professionally transcribed by Rev.com for benchmarking ASR on named-entity-dense speech.

- **Task:** asr
- **Languages:** en
- **Hours:** 39
- **Domain:** earnings call
- **License:** CC BY-SA 4.0
- **Homepage:** [https://github.com/revdotcom/speech-datasets](https://github.com/revdotcom/speech-datasets)
- **Paper:** [https://arxiv.org/abs/2104.11348](https://arxiv.org/abs/2104.11348)

## Recommendation

Use as an entity-dense ASR evaluation benchmark for long-form financial/business audio, especially when testing proper-noun and ticker handling. Small at 39h — best as an eval/probe set, not for large-scale training.

## Getting the data

Obtain from the [dataset homepage](https://github.com/revdotcom/speech-datasets).

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
