<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# GigaSpeech 2

Large-scale multi-domain ASR for low-resource Southeast Asian languages (Thai, Indonesian, Vietnamese), built by automated YouTube crawling and transcription (~30k h raw, ~22k h refined).

- **Task:** asr, multilingual
- **Languages:** th, id, vi
- **Hours:** 30000
- **Domain:** youtube
- **License:** see source terms
- **Homepage:** [https://github.com/SpeechColab/GigaSpeech2](https://github.com/SpeechColab/GigaSpeech2)
- **Paper:** [https://arxiv.org/abs/2406.11546](https://arxiv.org/abs/2406.11546)

## Recommendation

Pick this for low-resource SE Asian ASR where labeled data is scarce; the refined splits give usable labels plus professional dev/test sets. Audio is gated and restricted to non-commercial research/education, and labels are machine-generated/auto-refined, so quality varies by split.

## Getting the data

Obtain from the [dataset homepage](https://github.com/SpeechColab/GigaSpeech2).

HF tags Apache-2.0 but access is gated with non-commercial research/education terms; SpeechColab does not own the audio copyright.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
