<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# CREMA-D

7,442 acted audio-visual emotional clips from 91 demographically diverse actors speaking 12 sentences in 6 emotions at 4 intensity levels.

- **Task:** emotion, speaker
- **Languages:** en
- **Domain:** acted emotional
- **License:** ODbL 1.0 (database) + DbCL 1.0 (contents)
- **Homepage:** [https://github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- **Paper:** [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4313618/)

## Recommendation

Good for demographically diverse acted emotion recognition and audio-visual affect work, with crowd-sourced perceptual ratings included. Choose it when speaker/ethnic diversity matters; emotion is acted and the corpus is modest in size.

## Getting the data

Obtain from the [dataset homepage](https://github.com/CheyneyComputerScience/CREMA-D).

Openly available on GitHub under Open Data Commons licenses.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/emotion-recognize.yaml` — run it with `vkit docker run`.
