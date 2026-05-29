<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# TIMIT Acoustic-Phonetic Continuous Speech Corpus

630 American English speakers across 8 dialect regions, each reading 10 phonetically rich sentences, with time-aligned phonetic and word transcriptions.

- **Task:** asr
- **Languages:** en
- **Hours:** 5
- **Domain:** phonetic read
- **License:** see source terms
- **Homepage:** [https://catalog.ldc.upenn.edu/LDC93S1](https://catalog.ldc.upenn.edu/LDC93S1)

## Recommendation

Canonical benchmark for phonetic recognition, acoustic-phonetic studies, and dialect/phoneme analysis — pick for small-scale academic work where standard comparability matters. Avoid for modern large-scale ASR training (only ~5h, read, dated 1993, paid LDC license).

## Getting the data

Obtain from the [dataset homepage](https://catalog.ldc.upenn.edu/LDC93S1).

Paid LDC distribution (LDC93S1); DOI 10.35111/17gk-bn40.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
