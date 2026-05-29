<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# SEAME (Mandarin-English Code-Switching Speech Corpus)

~192 hours of spontaneous Mandarin-English code-switching conversations and interviews from 156 Singaporean and Malaysian speakers on everyday topics.

- **Task:** asr, multilingual
- **Languages:** multi
- **Hours:** 192
- **Domain:** code-switching conversational
- **License:** see source terms
- **Homepage:** [https://catalog.ldc.upenn.edu/LDC2015S04](https://catalog.ldc.upenn.edu/LDC2015S04)

## Recommendation

The de facto benchmark for Mandarin-English code-switching ASR — pick when you need intra-sentence code-switching with realistic Southeast Asian accents. Modest size by modern standards; accent distribution (Singapore/Malaysia) may not transfer to mainland-China Mandarin or US English.

## Getting the data

Obtain from the [dataset homepage](https://catalog.ldc.upenn.edu/LDC2015S04).

Paid LDC distribution (LDC2015S04); 16 kHz FLAC; UTF-8 transcripts with per-token language labels.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
