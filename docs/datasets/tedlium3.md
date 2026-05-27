<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# TED-LIUM 3

452-hour English ASR corpus of TED talks with manual and automatic transcriptions; suitable for lecture/talk domain ASR research.

- **Task:** asr
- **Languages:** en
- **Hours:** 452
- **Domain:** TED talks
- **License:** CC BY-NC-ND 3.0
- **Homepage:** [https://www.openslr.org/51](https://www.openslr.org/51)
- **Paper:** [https://arxiv.org/abs/1805.04699](https://arxiv.org/abs/1805.04699)

## Recommendation

Good choice for spontaneous (but well-articulated) English ASR, contrasting with the read-speech style of LibriSpeech. Non-commercial license. Use the SPH or WAV releases — the SPH format needs conversion. Useful for domain-shift experiments alongside LibriSpeech.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/51).

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
