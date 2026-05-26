<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# LibriTTS

High-fidelity (24 kHz) read English audiobooks derived from LibriSpeech, with normalised transcriptions; the standard baseline for English TTS.


- **Task:** tts
- **Languages:** en
- **Hours:** 585
- **Domain:** audiobook
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/60](https://www.openslr.org/60)
- **Paper:** [https://arxiv.org/abs/1904.02882](https://arxiv.org/abs/1904.02882)

## Recommendation

Best first choice for English TTS training. Audio is clean and well-segmented; train-clean-360 is a solid single-subset starting point. Use train-other-500 to add acoustic diversity. Lacks spontaneous or expressive speech — supplement with VCTK or Emilia for prosody range.


## Getting the data

Downloadable via VoxKitchen (`libritts`, source: openslr, size: 881 MB - 41.5 GB):

```bash
vkit docker download --tag slim libritts --root ./data/libritts
```

Subsets: dev-clean, dev-other, test-clean, test-other, train-clean-100, train-clean-360, train-other-500.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
