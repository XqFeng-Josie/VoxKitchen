<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Thorsten-Voice (German Neutral TTS)

A free single-speaker German read-speech corpus (~22,668 phrases, 22.05 kHz mono) recorded by Thorsten Müller for open TTS training, with neutral and emotional variants released over multiple years.

- **Task:** tts
- **Languages:** de
- **Hours:** 23
- **Domain:** read (home-studio, single male speaker)
- **License:** CC0-1.0
- **Homepage:** [https://www.thorsten-voice.de/en/datasets-2/](https://www.thorsten-voice.de/en/datasets-2/)

## Recommendation

The go-to open German TTS dataset — pick for any de-DE single-speaker pipeline where licensing must be unencumbered (used by Coqui, Piper, Home Assistant). Default to the 2021.02 Neutral release (23 h, CC0). Note sample rate is 22.05 kHz, not 24 kHz.

## Getting the data

Downloadable via VoxKitchen (`thorsten_voice`, source: openslr, size: 2.8 GB):

```bash
vkit docker download --tag slim thorsten_voice --root ./data/thorsten_voice
```

Subsets: default.

Mirrored on OpenSLR-95 (neutral) and OpenSLR-110 (emotional). The 2022.10 Zenodo release (DOI 10.5281/zenodo.7265581) is metadata-tagged CC BY 4.0 despite the project's CC0 mission — prefer the 2021.02 OpenSLR release for clean CC0 provenance.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
