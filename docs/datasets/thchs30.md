<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# THCHS-30 (Tsinghua Chinese 30-hour Database)

A 30-hour Mandarin read-speech corpus from CSLT Tsinghua, 16 kHz, with word/syllable/phone-level transcriptions and 50 speakers in a quiet office.

- **Task:** asr
- **Languages:** zh
- **Hours:** 30
- **Domain:** read
- **License:** Apache-2.0
- **Homepage:** [https://www.openslr.org/18/](https://www.openslr.org/18/)
- **Paper:** [https://arxiv.org/abs/1512.01882](https://arxiv.org/abs/1512.01882)

## Recommendation

Great lightweight baseline for Mandarin ASR experiments, recipe smoke-tests, and teaching/demo pipelines — small size, permissive Apache-2.0. Pair with AISHELL-1/2 or WenetSpeech for production scale.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/18/).

Distributed via OpenSLR mirrors; ~6.4 GB compressed.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
