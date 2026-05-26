<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# WenetSpeech

10,000-hour large-scale Mandarin ASR corpus collected from YouTube and podcasts with automatic labelling.


- **Task:** asr
- **Languages:** zh
- **Hours:** 10000
- **Domain:** in-the-wild (YouTube, podcasts, audiobooks)
- **License:** see source terms
- **Homepage:** [https://github.com/wenet-e2e/WenetSpeech](https://github.com/wenet-e2e/WenetSpeech)
- **Paper:** [https://arxiv.org/abs/2110.03370](https://arxiv.org/abs/2110.03370)

## Recommendation

Go-to corpus for large-scale Mandarin ASR where AISHELL-1 is too clean or too small. Automatic labels introduce noise — expect to filter with the data-cleaning pipeline. Non-commercial restrictions apply; check the source terms before production use.


## Getting the data

Obtain from the [dataset homepage](https://github.com/wenet-e2e/WenetSpeech).

Register for access and download via the WenetSpeech toolkit. The full corpus requires several TB of storage. Quality varies by subset — the "L" training set has the most automatic-label noise.


## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/data-cleaning.yaml` — run it with `vkit docker run`.
