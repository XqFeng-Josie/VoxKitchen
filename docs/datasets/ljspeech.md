<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# LJSpeech

Single-speaker English TTS corpus (24 h, 13,100 clips) recorded from LibriVox readings. Universally used as a single-speaker TTS baseline.


- **Task:** tts
- **Languages:** en
- **Hours:** 24
- **Domain:** audiobook
- **License:** Public Domain
- **Homepage:** [https://keithito.com/LJ-Speech-Dataset/](https://keithito.com/LJ-Speech-Dataset/)

## Recommendation

Ideal for single-speaker TTS experiments and for quickly checking a pipeline end-to-end due to its small size. Because it is a single speaker it is not suitable for multi-speaker or voice-cloning experiments. Prefer LibriTTS or VCTK when speaker diversity matters.


## Getting the data

Downloadable via VoxKitchen (`ljspeech`, source: keithito, size: 2.6 GB):

```bash
vkit docker download --tag slim ljspeech --root ./data/ljspeech
```

Subsets: default.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
