<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MagicData-RAMC (Rich Annotated Mandarin Conversational)

180 hours of Mandarin two-party conversational telephone-style speech from 663 speakers across Chinese accent regions, with speaker-turn and topic annotations spanning daily-life to technology topics.

- **Task:** asr, speaker
- **Languages:** zh
- **Hours:** 180
- **Domain:** conversational
- **License:** see source terms
- **Homepage:** [https://www.openslr.org/123/](https://www.openslr.org/123/)
- **Paper:** [https://arxiv.org/abs/2203.16844](https://arxiv.org/abs/2203.16844)

## Recommendation

Strong fit for conversational Mandarin ASR, speaker diarization, and turn-taking research where read-speech corpora fall short. Choose when you need spontaneous dialogue with speaker-attributed transcripts.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/123/).

OpenSLR distribution is CC BY-NC-ND 4.0 — research-only, non-commercial, no derivatives. Verify before integrating into derivative datasets.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
