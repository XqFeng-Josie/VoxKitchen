<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# SLUE (Spoken Language Understanding Evaluation)

English SLU benchmark on natural (not read) speech — Phase-1 adds ASR, named-entity recognition, and sentiment annotations over subsets of VoxPopuli and VoxCeleb; Phase-2 adds dialog act classification, QA, summarization, and named-entity localization.

- **Task:** asr
- **Languages:** en
- **Hours:** 27.3
- **Domain:** spoken language understanding
- **License:** see source terms
- **Homepage:** [https://asappresearch.github.io/slue-toolkit/](https://asappresearch.github.io/slue-toolkit/)
- **Paper:** [https://arxiv.org/abs/2111.10367](https://arxiv.org/abs/2111.10367)

## Recommendation

Use SLUE to benchmark end-to-end and pipeline SLU systems on real-world English speech with consistent annotations — Phase-1 for short-utterance NER/SA/ASR, Phase-2 for longer-form discourse tasks. Licensing is inherited from upstream VoxPopuli (CC0) and VoxCeleb (research-only), so downstream redistribution must respect both source terms.

## Getting the data

Obtain from the [dataset homepage](https://asappresearch.github.io/slue-toolkit/).

Phase-1 fine-tune split totals 27.3h (VoxPopuli 14.5h + VoxCeleb 12.8h). Phase-2 paper at https://arxiv.org/abs/2212.10525. Toolkit (MIT) at https://github.com/asappresearch/slue-toolkit; HF datasets `asapp/slue` and `asapp/slue-phase-2`.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
