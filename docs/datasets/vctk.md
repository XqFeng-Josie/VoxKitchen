<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# VCTK

44-hour English multi-speaker corpus with 110 speakers covering a wide range of UK and US accents; widely used for multi-speaker TTS and speaker adaptation research.


- **Task:** tts, speaker
- **Languages:** en
- **Hours:** 44
- **Domain:** read sentences (accent-diverse)
- **License:** CC BY 4.0
- **Homepage:** [https://datashare.ed.ac.uk/handle/10283/3443](https://datashare.ed.ac.uk/handle/10283/3443)

## Recommendation

First choice for multi-speaker English TTS experiments and accent-aware speaker embedding research. The diversity of accents is its main advantage over LJSpeech. Recording quality is very clean. Hours per speaker are limited (~30 min), which constrains voice-cloning fine-tuning.


## Getting the data

Obtain from the [dataset homepage](https://datashare.ed.ac.uk/handle/10283/3443).

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
