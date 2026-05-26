<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# VoxCeleb2

2,442-hour large-scale speaker recognition corpus with 6,112 celebrities collected from YouTube across many languages.


- **Task:** speaker
- **Languages:** multi
- **Hours:** 2442
- **Domain:** celebrity speech (in-the-wild)
- **License:** see source terms
- **Homepage:** [https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)
- **Paper:** [https://arxiv.org/abs/1806.05622](https://arxiv.org/abs/1806.05622)

## Recommendation

The standard pre-training corpus for speaker verification models. Large scale and in-the-wild diversity make it near-mandatory for production speaker embeddings. Academic use only — commercial use requires a separate agreement. Download requires requesting access via the VoxCeleb website.


## Getting the data

Obtain from the [dataset homepage](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).

Download requires registration and acceptance of terms at the VoxCeleb website. Audio must be downloaded from YouTube using the provided scripts.


## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
