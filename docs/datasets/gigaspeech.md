<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# GigaSpeech

10,000-hour multi-domain English ASR corpus spanning audiobooks, podcasts, and YouTube.

- **Task:** asr
- **Languages:** en
- **Hours:** 10000
- **Domain:** audiobook, podcast, youtube
- **License:** see source terms
- **Homepage:** [https://github.com/SpeechColab/GigaSpeech](https://github.com/SpeechColab/GigaSpeech)

## Recommendation

Choose when you need scale and domain diversity beyond LibriSpeech (podcasts, YouTube). Requires accepting terms and an access request, and is large — plan storage carefully. No VoxKitchen recipe; download manually then point a dir-ingest ASR pipeline at your local copy.

## Getting the data

Obtain from the [dataset homepage](https://github.com/SpeechColab/GigaSpeech).

Request access via the GigaSpeech GitHub repository and follow the download instructions there. The XL split (10,000 h) requires ~1 TB of storage; the M/L splits are more manageable for prototyping.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
