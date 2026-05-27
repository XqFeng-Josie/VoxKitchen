<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Golos

~1,240 h of manually annotated open Russian speech split between crowd-sourced (~1,106 h) and farfield/smart-device (~134 h) recordings.

- **Task:** asr
- **Languages:** ru
- **Hours:** 1240
- **Domain:** crowdsourced + farfield
- **License:** see source terms
- **Homepage:** [https://github.com/sberdevices/golos](https://github.com/sberdevices/golos)
- **Paper:** [https://arxiv.org/abs/2106.10161](https://arxiv.org/abs/2106.10161)

## Recommendation

A strong default for Russian ASR training and benchmarking, especially for farfield/voice-assistant conditions alongside clean crowd speech. Licensing is a custom Sber document (not a standard identifier) — review before commercial use.

## Getting the data

Obtain from the [dataset homepage](https://github.com/sberdevices/golos).

License is the Sber document in the repo; also mirrored on OpenSLR #114 and HuggingFace (SberDevices/Golos).

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
