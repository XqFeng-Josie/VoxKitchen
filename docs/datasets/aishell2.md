<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AISHELL-2

1000 hours of clean Mandarin read-speech from ~1991 speakers covering entertainment, finance, technology, sports, and place-of-interest commands, recorded over iOS/Android/microphone channels.

- **Task:** asr
- **Languages:** zh
- **Hours:** 1000
- **Domain:** read
- **License:** see source terms
- **Homepage:** [http://www.aishelltech.com/aishell_2](http://www.aishelltech.com/aishell_2)
- **Paper:** [https://arxiv.org/abs/1808.10583](https://arxiv.org/abs/1808.10583)

## Recommendation

A standard industrial-scale Mandarin ASR training set — pick it for a serious read-speech baseline beyond AISHELL-1 or THCHS-30. Distribution is gated (free for academic use after a signed agreement with Shell Shell Technology); not redistributable.

## Getting the data

Obtain from the [dataset homepage](http://www.aishelltech.com/aishell_2).

Apply to AISHELL/Shell Shell with an institutional agreement. iOS channel is the free academic release; Android/Mic channels are dev/test only.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
