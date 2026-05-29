<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# VoxCeleb1

Speaker identification/verification corpus of 153,516 utterances from 1251 celebrities extracted from YouTube interview videos.

- **Task:** speaker
- **Languages:** multi
- **Hours:** 352
- **Domain:** celebrity interviews (YouTube)
- **License:** CC BY-SA 4.0
- **Homepage:** [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)
- **Paper:** [https://arxiv.org/abs/1706.08612](https://arxiv.org/abs/1706.08612)

## Recommendation

Default in-the-wild speaker-verification benchmark — pick when you want a public, well-comparable baseline before scaling to VoxCeleb2. Speakers are multi-nationality (speech predominantly English); YouTube provenance means clip-level quality varies and some links rot.

## Getting the data

Obtain from the [dataset homepage](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html).

Same VGG distribution model as VoxCeleb2; requires accepting the access terms on the VGG site even though the corpus itself is CC BY-SA 4.0.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
