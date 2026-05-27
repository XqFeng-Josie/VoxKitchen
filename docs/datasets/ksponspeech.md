<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# KsponSpeech

~969 h of Korean spontaneous open-domain dialogue from ~2,000 native speakers, with dual orthographic + pronunciation transcription.

- **Task:** asr
- **Languages:** ko
- **Hours:** 969
- **Domain:** conversational
- **License:** see source terms
- **Homepage:** [https://aihub.or.kr/aidata/105](https://aihub.or.kr/aidata/105)
- **Paper:** [https://www.mdpi.com/2076-3417/10/19/6936](https://www.mdpi.com/2076-3417/10/19/6936)

## Recommendation

The standard large-scale corpus for Korean spontaneous-speech ASR and the reference for Korean ASR toolkits. Distributed via the Korean government AIHub portal under custom terms requiring registration/approval, which can be a barrier for non-Korean users.

## Getting the data

Obtain from the [dataset homepage](https://aihub.or.kr/aidata/105).

Requires an AIHub account and agreement to AIHub usage terms; not a standard open license.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
