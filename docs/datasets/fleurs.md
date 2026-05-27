<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# FLEURS

Few-shot Learning Evaluation of Universal Representations of Speech — standardised ASR/LID evaluation set covering 102 languages derived from the FLoRes-200 text corpus.

- **Task:** asr, multilingual
- **Languages:** multi
- **Domain:** read speech (FLoRes translations)
- **License:** CC BY 4.0
- **Homepage:** [https://huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs)
- **Paper:** [https://arxiv.org/abs/2205.12446](https://arxiv.org/abs/2205.12446)

## Recommendation

The standard multilingual ASR evaluation benchmark. Use it to measure cross-lingual ASR quality consistently across languages rather than as a training corpus (each language only has ~10 h). Available via HuggingFace datasets — VoxKitchen's recipe handles the streaming download.

## Getting the data

Downloadable via VoxKitchen (`fleurs`, source: HuggingFace, size: —):

```bash
vkit docker download --tag slim fleurs --root ./data/fleurs
```

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/fleurs-multilingual.yaml` — run it with `vkit docker run`.
