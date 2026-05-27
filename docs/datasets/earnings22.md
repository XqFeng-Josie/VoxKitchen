<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Earnings-22

119 h benchmark of real-world English corporate earnings calls featuring diverse global accents across many countries.

- **Task:** asr
- **Languages:** en
- **Hours:** 119
- **Domain:** earnings calls
- **License:** CC BY-SA 4.0
- **Homepage:** [https://github.com/revdotcom/speech-datasets/tree/main/earnings22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22)
- **Paper:** [https://arxiv.org/abs/2203.15591](https://arxiv.org/abs/2203.15591)

## Recommendation

Use as an evaluation benchmark for accented, real-world long-form English ASR rather than a large training set — ideal to stress-test robustness across non-native and regional accents. Small (~125 files) and benchmark-oriented, with rich per-file metadata.

## Getting the data

Obtain from the [dataset homepage](https://github.com/revdotcom/speech-datasets/tree/main/earnings22).

Released by Rev.com; speakers span 7 language regions / 27 countries.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
