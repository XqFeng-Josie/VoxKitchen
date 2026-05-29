<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# IMDA National Speech Corpus (NSC)

Large-scale Singapore-English speech corpus from IMDA — ~2000 hours of orthographically transcribed read speech plus ~1000 hours of conversational speech, designed for ASR research on Singapore-accented English.

- **Task:** asr
- **Languages:** en
- **Hours:** 3000
- **Domain:** read + conversational accented
- **License:** see source terms
- **Homepage:** [https://www.imda.gov.sg/how-we-can-help/national-speech-corpus](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus)
- **Paper:** [https://www.isca-archive.org/interspeech_2019/koh19_interspeech.html](https://www.isca-archive.org/interspeech_2019/koh19_interspeech.html)

## Recommendation

Top pick for Singapore-English / Southeast-Asian-accented ASR training and adaptation, and one of the largest openly available accented-English corpora. Choose when you need locally-relevant vocabulary, code-mixing patterns, or non-US/UK English coverage.

## Getting the data

Obtain from the [dataset homepage](https://www.imda.gov.sg/how-we-can-help/national-speech-corpus).

Distributed under IMDA's licence (often referenced as Singapore Open Data Licence v1.0); request-based access via nsc@imda.gov.sg; multi-part release (Parts 1-6) with substantial storage requirements.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
