<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AISHELL-3

85-hour multi-speaker Mandarin TTS corpus with 218 speakers in clean recording conditions; the standard Chinese multi-speaker TTS baseline.


- **Task:** tts
- **Languages:** zh
- **Hours:** 85
- **Domain:** read speech
- **License:** CC BY-NC-ND 4.0
- **Homepage:** [https://www.openslr.org/93](https://www.openslr.org/93)
- **Paper:** [https://arxiv.org/abs/2010.11567](https://arxiv.org/abs/2010.11567)

## Recommendation

Best starting point for multi-speaker Mandarin TTS. Clean studio conditions and a large speaker pool make it suitable for voice cloning research. Non-commercial license — check before production use.


## Getting the data

Downloadable via VoxKitchen (`aishell3`, source: openslr, size: 17.7 GB):

```bash
vkit docker download --tag slim aishell3 --root ./data/aishell3
```

Subsets: data_aishell3.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
