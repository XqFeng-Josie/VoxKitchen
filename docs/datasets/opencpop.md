<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Opencpop

High-quality Mandarin singing-voice synthesis corpus of 100 popular Chinese pop songs (3756 utterances) sung by a single female professional vocalist, 44.1 kHz, with phoneme/note boundary and pitch annotations.

- **Task:** tts
- **Languages:** zh
- **Hours:** 5.2
- **Domain:** mandarin pop singing
- **License:** CC BY-NC-ND 4.0
- **Homepage:** [https://wenet-e2e.github.io/opencpop/](https://wenet-e2e.github.io/opencpop/)
- **Paper:** [https://arxiv.org/abs/2201.07429](https://arxiv.org/abs/2201.07429)

## Recommendation

Default open benchmark for Mandarin singing-voice synthesis (SVS) — pick for SVS prototyping, F0/note-conditional TTS, and karaoke experiments. Single-singer female vocal limits speaker generalization, and the CC BY-NC-ND license blocks commercial use and derivative redistribution.

## Getting the data

Obtain from the [dataset homepage](https://wenet-e2e.github.io/opencpop/).

Non-commercial, no-derivatives license; single-singer corpus.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
