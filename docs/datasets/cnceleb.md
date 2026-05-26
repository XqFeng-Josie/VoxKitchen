<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# CN-Celeb

1,200-hour multi-genre Mandarin speaker recognition corpus spanning 11 real-world scenarios collected from Chinese celebrities.


- **Task:** speaker
- **Languages:** zh
- **Hours:** 1200
- **Domain:** celebrity speech (in-the-wild)
- **License:** see source terms
- **Homepage:** [https://www.openslr.org/82](https://www.openslr.org/82)
- **Paper:** [https://arxiv.org/abs/1911.01799](https://arxiv.org/abs/1911.01799)

## Recommendation

The primary benchmark for Mandarin speaker verification and identification in realistic, in-the-wild conditions. Wide acoustic diversity across genres (interview, singing, entertainment) is valuable but also makes it challenging. Check the source terms carefully — the data is free for research but redistribution is restricted.


## Getting the data

Downloadable via VoxKitchen (`cnceleb`, source: openslr, size: 20.7 GB):

```bash
vkit docker download --tag slim cnceleb --root ./data/cnceleb
```

Subsets: cn-celeb_v2.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
