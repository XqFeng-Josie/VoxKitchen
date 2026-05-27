<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Emotional Speech Database (ESD)

>29 h of parallel emotional speech from 20 speakers (10 English, 10 Mandarin), each reading 350 parallel utterances across 5 emotions.

- **Task:** tts, emotion
- **Languages:** en, zh
- **Hours:** 29
- **Domain:** acted emotional (parallel, bilingual)
- **License:** see source terms
- **Homepage:** [https://hltsingapore.github.io/ESD/](https://hltsingapore.github.io/ESD/)
- **Paper:** [https://arxiv.org/abs/2105.14762](https://arxiv.org/abs/2105.14762)

## Recommendation

The go-to corpus for emotional voice conversion and cross-lingual / multi-speaker emotional TTS thanks to its parallel bilingual design. Emotion is acted, only 5 classes, and 20 speakers limit speaker diversity.

## Getting the data

Obtain from the [dataset homepage](https://hltsingapore.github.io/ESD/).

Released by NUS/SUTD for research use; the project page gives no formal license, only a citation request — treat as research-only.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
