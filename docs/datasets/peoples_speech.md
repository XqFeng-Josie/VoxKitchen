<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# People's Speech

30,000-hour English ASR corpus assembled from diverse internet sources including radio broadcasts, court hearings, and conferences.


- **Task:** asr
- **Languages:** en
- **Hours:** 30000
- **Domain:** diverse (radio, court, conference, podcast)
- **License:** CC BY-SA 4.0
- **Homepage:** [https://huggingface.co/datasets/MLCommons/peoples_speech](https://huggingface.co/datasets/MLCommons/peoples_speech)
- **Paper:** [https://arxiv.org/abs/2111.09344](https://arxiv.org/abs/2111.09344)

## Recommendation

A large permissively-licensed English corpus that adds domain diversity beyond audiobooks. Automatic labels vary in quality — apply the data-cleaning pipeline to filter low-confidence segments before training. Good complement to LibriSpeech/GigaSpeech in a multi-dataset training mix.


## Getting the data

Obtain from the [dataset homepage](https://huggingface.co/datasets/MLCommons/peoples_speech).

Available via HuggingFace datasets. The full 30,000 h "dirty" split has more noise; use the "clean" split or apply quality filtering for best ASR training results.


## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/data-cleaning.yaml` — run it with `vkit docker run`.
