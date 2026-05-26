<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AISHELL-1

170-hour open Mandarin speech corpus recorded in clean studio conditions; the standard Chinese ASR benchmark.


- **Task:** asr
- **Languages:** zh
- **Hours:** 170
- **Domain:** read speech
- **License:** CC BY-NC-ND 4.0
- **Homepage:** [https://www.openslr.org/33](https://www.openslr.org/33)
- **Paper:** [https://arxiv.org/abs/1709.05522](https://arxiv.org/abs/1709.05522)

## Recommendation

The go-to starting point for Mandarin ASR. Clean studio recordings with full transcripts. Non-commercial license — check before production use. Not representative of spontaneous or accented Mandarin; supplement with WenetSpeech for broader coverage.


## Getting the data

Downloadable via VoxKitchen (`aishell`, source: openslr, size: 1 MB - 14.5 GB):

```bash
vkit docker download --tag slim aishell --root ./data/aishell
```

Subsets: data_aishell, resource_aishell.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/asr-training-data.yaml` — run it with `vkit docker run`.
