<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MUSAN

109-hour corpus of music, speech, and environmental noise designed for data augmentation in speech and speaker recognition experiments.


- **Task:** augmentation
- **Languages:** en
- **Hours:** 109
- **Domain:** music, speech, noise
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/17](https://www.openslr.org/17)
- **Paper:** [https://arxiv.org/abs/1510.08484](https://arxiv.org/abs/1510.08484)

## Recommendation

The standard noise/music augmentation bank for speaker verification and robust ASR training. Combine with RIR-based room simulation for a full augmentation pipeline. Small enough to keep fully in RAM during training. No VoxKitchen pipeline template — augmentation is embedded in task-specific pipelines such as examples/pipelines/noise-augment.yaml.


## Getting the data

Downloadable via VoxKitchen (`musan`, source: openslr, size: 10.3 GB):

```bash
vkit docker download --tag slim musan --root ./data/musan
```

Subsets: musan.
