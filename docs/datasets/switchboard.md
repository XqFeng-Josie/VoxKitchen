<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Switchboard-1 Release 2

~2,400 two-sided spontaneous English telephone conversations among 543 US speakers (~260 h), separated into two channels.

- **Task:** asr, speaker
- **Languages:** en
- **Hours:** 260
- **Domain:** conversational telephone
- **License:** see source terms
- **Homepage:** [https://catalog.ldc.upenn.edu/LDC97S62](https://catalog.ldc.upenn.edu/LDC97S62)

## Recommendation

A classic benchmark for conversational telephone ASR (Hub5/SWBD) and speaker recognition; choose it for spontaneous narrowband (8 kHz) two-party dialogue or to compare against decades of published WER. It is a paid LDC corpus, and telephone-bandwidth audio differs from modern wideband data.

## Getting the data

Obtain from the [dataset homepage](https://catalog.ldc.upenn.edu/LDC97S62).

Requires an LDC license / paid membership (catalog ID LDC97S62); not openly downloadable.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
