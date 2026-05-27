<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Libri-Light

~60k h of unlabelled English read speech from LibriVox audiobooks, with small labelled subsets (10h, 1h, 10min) for limited-supervision ASR.

- **Task:** asr
- **Languages:** en
- **Hours:** 60000
- **Domain:** audiobook
- **License:** public domain (LibriVox)
- **Homepage:** [https://github.com/facebookresearch/libri-light](https://github.com/facebookresearch/libri-light)
- **Paper:** [https://arxiv.org/abs/1912.07875](https://arxiv.org/abs/1912.07875)

## Recommendation

The standard corpus for self-supervised and low-resource English ASR and for reproducing the 10h/1h/10min limited-supervision benchmarks. Read audiobook speech only, so it is poor for conversational or multilingual targets.

## Getting the data

Obtain from the [dataset homepage](https://github.com/facebookresearch/libri-light).

Underlying LibriVox audio is public domain; the dataset tooling/repo is MIT-licensed.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
