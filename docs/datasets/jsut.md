<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# JSUT (Japanese speech corpus of Saruwatari-lab, U-Tokyo)

A ~10-hour single-speaker Japanese read-speech corpus designed for end-to-end TTS, covering the main pronunciations of daily-use Japanese characters.

- **Task:** tts
- **Languages:** ja
- **Hours:** 10
- **Domain:** read (studio, single female speaker)
- **License:** see source terms
- **Homepage:** [https://sites.google.com/site/shinnosuketakamichi/publication/jsut](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
- **Paper:** [https://arxiv.org/abs/1711.00354](https://arxiv.org/abs/1711.00354)

## Recommendation

Strong default for Japanese single-speaker TTS baselines and research prototypes — the canonical reference corpus alongside JVS. Pick when you need a clean, well-known ja-JP voice for studio-quality TTS.

## Getting the data

Obtain from the [dataset homepage](https://sites.google.com/site/shinnosuketakamichi/publication/jsut).

License is split — text is CC BY-SA 4.0, audio is research-only by default; commercial audio use requires emailing Takamichi / U-Tokyo TLO for permission.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
