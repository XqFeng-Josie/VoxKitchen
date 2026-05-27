<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Hi-Fi Multi-Speaker English TTS (Hi-Fi TTS)

~291.6 h high-quality English multi-speaker TTS from 10 LibriVox speakers (>=17 h each), 44.1 kHz, with Project Gutenberg text.

- **Task:** tts
- **Languages:** en
- **Hours:** 292
- **Domain:** audiobook
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/109/](https://www.openslr.org/109/)
- **Paper:** [https://arxiv.org/abs/2104.01497](https://arxiv.org/abs/2104.01497)

## Recommendation

A strong choice for high-fidelity English multi-speaker or single-speaker TTS — clean 44.1 kHz audio with generous per-speaker hours, and CC BY 4.0 / public-domain sourcing safe for commercial use. Prefer it over CSS10 for English. Distinct from the much larger HiFiTTS-2 (~36.7k h).

## Getting the data

Obtain from the [dataset homepage](https://www.openslr.org/109/).

NVIDIA release on OpenSLR (SLR109). Do not confuse with HiFiTTS-2.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
