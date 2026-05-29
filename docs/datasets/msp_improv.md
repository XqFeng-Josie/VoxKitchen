<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MSP-IMPROV

Acted dyadic emotional speech corpus from UT Dallas with 12 actors across six dyad sessions producing 8438 speaking turns (652 target sentences) labelled for happiness, sadness, anger, and neutral.

- **Task:** emotion, speaker
- **Languages:** en
- **Domain:** dyadic improvised
- **License:** see source terms
- **Homepage:** [https://lab-msp.com/MSP/MSP-Improv.html](https://lab-msp.com/MSP/MSP-Improv.html)
- **Paper:** [https://ieeexplore.ieee.org/document/7374697/](https://ieeexplore.ieee.org/document/7374697/)

## Recommendation

Use for emotion recognition where you want controlled lexical content but more naturalistic delivery than fully scripted corpora, and for studying perception/elicitation strategies. Good companion to IEMOCAP for cross-corpus generalization. Small speaker pool (12), acted (not in-the-wild).

## Getting the data

Obtain from the [dataset homepage](https://lab-msp.com/MSP/MSP-Improv.html).

Academic access requires a signed agreement with Prof. Carlos Busso; commercial license is paid (priced at US$8000 per the official page). Total hours not officially published.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/emotion-recognize.yaml` — run it with `vkit docker run`.
