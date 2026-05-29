<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MELD (Multimodal EmotionLines Dataset)

Multimodal (audio, video, text) emotion recognition corpus of ~13k utterances from ~1.4k multi-party dialogues sampled from the Friends TV series, labelled with seven emotions and three-way sentiment.

- **Task:** emotion, speaker
- **Languages:** en
- **Hours:** 13
- **Domain:** tv dialog
- **License:** GPL-3.0
- **Homepage:** [https://affective-meld.github.io](https://affective-meld.github.io)
- **Paper:** [https://aclanthology.org/P19-1050/](https://aclanthology.org/P19-1050/)

## Recommendation

Strong choice for benchmarking emotion recognition in conversation, especially multi-party dialog with speaker turns; useful when you need aligned audio+text+visual modalities. TV-acted English only, modest scale (~13 h), class imbalance (Neutral dominant); audio carries music/laugh-track artifacts from broadcast tracks.

## Getting the data

Obtain from the [dataset homepage](https://affective-meld.github.io).

Distributed via the declare-lab/MELD GitHub repo; raw .mp4 splits are downloadable without gating. GPL-3.0 license applies to the corpus.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/emotion-recognize.yaml` — run it with `vkit docker run`.
