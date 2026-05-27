<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AISHELL-4

Real-recorded Mandarin conference-meeting corpus (8-channel circular mic array), 211 sessions with 4-8 speakers each, annotated for transcription and speaker activity.

- **Task:** asr, speaker
- **Languages:** zh
- **Hours:** 120
- **Domain:** meetings
- **License:** CC BY-SA 4.0
- **Homepage:** [https://www.openslr.org/111/](https://www.openslr.org/111/)
- **Paper:** [https://arxiv.org/abs/2104.03603](https://arxiv.org/abs/2104.03603)

## Recommendation

Choose for Mandarin meeting/conference scenarios needing realistic far-field, multi-speaker, overlapping speech — supports ASR, diarization, separation, and enhancement. 120 h is modest and array-based far-field, so it suits meeting-domain work more than general ASR.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/111/).

211 sessions, 8-channel circular mic array; distributed via OpenSLR

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
