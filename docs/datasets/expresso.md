<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Expresso

High-quality multi-speaker English expressive speech at 48 kHz (11 h read + 30 h improvised) across many spontaneous expressive styles, for expressive speech resynthesis.

- **Task:** tts, emotion
- **Languages:** en
- **Hours:** 40
- **Domain:** expressive read + improvised studio
- **License:** CC BY-NC 4.0
- **Homepage:** [https://speechbot.github.io/expresso/](https://speechbot.github.io/expresso/)
- **Paper:** [https://arxiv.org/abs/2308.05725](https://arxiv.org/abs/2308.05725)

## Recommendation

Best for expressive/style-controlled TTS and discrete speech-resynthesis research where studio-clean, style-labeled English audio matters. Only 4 speakers and a non-commercial license, so it is unsuitable for commercial training and limited for speaker-diversity work.

## Getting the data

Obtain from the [dataset homepage](https://speechbot.github.io/expresso/).

Distributed via Meta's facebookresearch/textlesslib repo; NonCommercial.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
