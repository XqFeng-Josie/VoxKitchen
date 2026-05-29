<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MGB-2 Challenge (Arabic Multi-Dialect Broadcast Media Recognition)

1200 hours of lightly supervised Arabic broadcast speech from 19 Al Jazeera Arabic TV programmes (2005-2015) — conversations, interviews, reports — with multi-dialect coverage.

- **Task:** asr, multilingual
- **Languages:** ar
- **Hours:** 1200
- **Domain:** Arabic TV broadcast (Al Jazeera)
- **License:** see source terms
- **Homepage:** [http://www.mgb-challenge.org/MGB-2.html](http://www.mgb-challenge.org/MGB-2.html)
- **Paper:** [https://arxiv.org/abs/1609.05625](https://arxiv.org/abs/1609.05625)

## Recommendation

The strongest publicly known Arabic broadcast ASR resource — pick for Arabic ASR or multi-dialect acoustic modeling at scale. Transcriptions are lightly supervised (not gold) and access is gated through the MGB organizers (QCRI); dialect labels are not exhaustive.

## Getting the data

Obtain from the [dataset homepage](http://www.mgb-challenge.org/MGB-2.html).

Includes ~110M-word LM text from aljazeera.net.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
