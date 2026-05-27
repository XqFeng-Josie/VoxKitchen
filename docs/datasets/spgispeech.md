<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# SPGISpeech

5,000 h of professionally transcribed English company earnings-call audio, fully formatted with punctuation and capitalization.

- **Task:** asr
- **Languages:** en
- **Hours:** 5000
- **Domain:** earnings calls
- **License:** see source terms
- **Homepage:** [https://datasets.kensho.com/datasets/spgispeech](https://datasets.kensho.com/datasets/spgispeech)
- **Paper:** [https://arxiv.org/abs/2104.02014](https://arxiv.org/abs/2104.02014)

## Recommendation

A strong pick for English ASR on spontaneous, accented, real-world business speech with high-quality fully-formatted (punctuated, denormalized) transcripts. Access is gated behind a Kensho research agreement, so it is not freely redistributable.

## Getting the data

Obtain from the [dataset homepage](https://datasets.kensho.com/datasets/spgispeech).

Gated: requires signing the Kensho research download agreement. A newer SPGISpeech 2.0 (~3,780 h, speaker-tagged) also exists.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
