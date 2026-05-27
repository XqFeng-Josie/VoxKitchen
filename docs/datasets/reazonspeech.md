<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# ReazonSpeech

~35,000 h open Japanese speech corpus collected from terrestrial TV broadcast streams with aligned Japanese transcriptions.

- **Task:** asr
- **Languages:** ja
- **Hours:** 35000
- **Domain:** tv/broadcast
- **License:** CDLA-Sharing-1.0
- **Homepage:** [https://research.reazon.jp/projects/ReazonSpeech/](https://research.reazon.jp/projects/ReazonSpeech/)
- **Paper:** [https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf](https://research.reazon.jp/_static/reazonspeech_nlp2023.pdf)

## Recommendation

The best choice for large-scale Japanese ASR pretraining or fine-tuning, given its scale and natural broadcast speech. Use is legally constrained to Japanese Copyright Act Art. 30-4 (text/data-mining R&D), so commercial deployment terms are restrictive; the dataset is gated.

## Getting the data

Obtain from the [dataset homepage](https://research.reazon.jp/projects/ReazonSpeech/).

HF dataset gated by agreement to use solely under Japanese Copyright Act Art. 30-4; sizes range 8.5 h to 35,000 h. Creation toolkit is Apache-2.0.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
