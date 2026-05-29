<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# LibriTTS-R

A sound-quality-restored version of LibriTTS — 585 hours of 24 kHz English read speech from 2456 speakers, identical samples/texts to LibriTTS but enhanced via Google's Miipher speech restoration model.

- **Task:** tts
- **Languages:** en
- **Hours:** 585
- **Domain:** audiobook (restored, multi-speaker read)
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/141/](https://www.openslr.org/141/)
- **Paper:** [https://arxiv.org/abs/2305.18802](https://arxiv.org/abs/2305.18802)

## Recommendation

The current default large-scale multi-speaker English TTS corpus — pick over plain LibriTTS whenever audio fidelity matters (zero-shot TTS, neural codec training, voice cloning baselines). Drop-in compatible with any LibriTTS pipeline; same speaker IDs, splits, and transcripts.

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/141/).

Still derived from LibriVox audiobooks, so prosody is narrative/literary; expect domain shift if your target is conversational.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
