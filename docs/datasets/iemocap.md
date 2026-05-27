<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# IEMOCAP

~12 h of acted audio-visual dyadic interactions from 10 actors (scripted and improvised), with categorical and dimensional (valence/activation/ dominance) emotion labels.

- **Task:** emotion, speaker
- **Languages:** en
- **Hours:** 12
- **Domain:** scripted dyadic
- **License:** see source terms
- **Homepage:** [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)
- **Paper:** [https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)

## Recommendation

A foundational benchmark for conversational/dyadic emotion recognition and multimodal affect modeling — choose it when you need dialog context and both categorical and dimensional labels. Emotion is acted, access is gated behind a signed license, and class distributions are imbalanced.

## Getting the data

Obtain from the [dataset homepage](https://sail.usc.edu/iemocap/).

Requires a signed academic release form from USC SAIL; not freely downloadable.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/emotion-recognize.yaml` — run it with `vkit docker run`.
