<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# Emilia

Large-scale multilingual in-the-wild speech dataset designed for expressive and diverse TTS training, covering 6 languages.

- **Task:** tts, multilingual
- **Languages:** multi
- **Domain:** in-the-wild (diverse)
- **License:** see source terms
- **Homepage:** [https://huggingface.co/datasets/amphion/Emilia-Dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset)
- **Paper:** [https://arxiv.org/abs/2407.05361](https://arxiv.org/abs/2407.05361)

## Recommendation

Best choice when you need expressive, diverse multi-lingual TTS training data that goes beyond clean audiobook recordings. Collected from in-the-wild audio so prosody and speaking style are varied — ideal for natural-sounding TTS. Check the source terms; access requires registration.

## Getting the data

Obtain from the [dataset homepage](https://huggingface.co/datasets/amphion/Emilia-Dataset).

Request access via the HuggingFace repository page. The dataset is available in processed (Emilia) and unprocessed (Emilia-Pipe) variants; use the processed variant unless you are doing your own quality filtering.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/tts-data-prep.yaml` — run it with `vkit docker run`.
