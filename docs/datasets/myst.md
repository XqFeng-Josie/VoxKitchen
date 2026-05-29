<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MyST Children's Conversational Speech

~470 hours of English conversational speech from 1371 students in grades 3-5 interacting with a virtual science tutor across eight FOSS-curriculum science topics, produced by Boulder Learning.

- **Task:** asr
- **Languages:** en
- **Hours:** 470
- **Domain:** children grades 3-5 STEM tutoring
- **License:** see source terms
- **Homepage:** [https://catalog.ldc.upenn.edu/LDC2021S05](https://catalog.ldc.upenn.edu/LDC2021S05)

## Recommendation

Pick for child-speech ASR where conversational, open-ended tutoring dialogue is needed — one of the largest English children's-speech corpora available. Caveats — paid LDC distribution, only ~45% of utterances are transcribed, and the grade range is 3-5 (not K-2).

## Getting the data

Obtain from the [dataset homepage](https://catalog.ldc.upenn.edu/LDC2021S05).

Distributed via LDC under the MyST Children's Conversational Speech Agreement; commercial use requires contacting Boulder Learning Inc.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
