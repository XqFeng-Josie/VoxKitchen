<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# CSS10

Single-speaker speech datasets for 10 languages built from aligned public-domain LibriVox clips, intended for TTS.

- **Task:** tts
- **Languages:** de, el, es, fi, fr, hu, ja, nl, ru, zh
- **Hours:** 99
- **Domain:** audiobook
- **License:** see source terms
- **Homepage:** [https://github.com/Kyubyong/css10](https://github.com/Kyubyong/css10)
- **Paper:** [https://arxiv.org/abs/1903.11269](https://arxiv.org/abs/1903.11269)

## Recommendation

A lightweight starting point for non-English single-speaker TTS, useful as a baseline or for low-resource prototyping. Per-language coverage is uneven (~10-20 h each), so it suits small or fine-tuning experiments. The license is reported inconsistently across sources — verify before redistribution.

## Getting the data

Obtain from the [dataset homepage](https://github.com/Kyubyong/css10).

Underlying LibriVox audio is public domain, but the repo declares Apache-2.0 and some mirrors label it CC BY-SA 4.0; confirm before commercial use.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
