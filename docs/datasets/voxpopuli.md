<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# VoxPopuli

Multilingual corpus from 2009-2020 European Parliament recordings: a large unlabelled set across 23 languages plus transcribed speech and aligned interpretations.

- **Task:** asr, multilingual
- **Languages:** multi
- **Domain:** parliament
- **License:** CC0
- **Homepage:** [https://github.com/facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli)
- **Paper:** [https://arxiv.org/abs/2101.00390](https://arxiv.org/abs/2101.00390)

## Recommendation

Best for multilingual self-supervised pretraining and European-language ASR fine-tuning in a formal/parliamentary register. The unlabelled vs transcribed subsets differ enormously in size, so total hours depend on which subset you ingest. Formal political speech transfers poorly to conversational or noisy audio.

## Getting the data

Obtain from the [dataset homepage](https://github.com/facebookresearch/voxpopuli).

CC0 release; raw audio subject to the European Parliament legal notice. Mirrored at huggingface.co/datasets/facebook/voxpopuli.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
