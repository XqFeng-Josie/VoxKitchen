<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# DiPCo (Dinner Party Corpus)

English far-field conversational corpus of 10 dinner-party sessions (4 participants each, 15-45 minutes per session) recorded with one close-talk microphone plus five 7-mic far-field array devices, designed for noise-robust distant ASR and diarization.

- **Task:** asr, speaker
- **Languages:** en
- **Domain:** far-field dinner party
- **License:** CDLA-Permissive-1.0
- **Homepage:** [https://www.amazon.science/publications/dipco-dinner-party-corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus)
- **Paper:** [https://arxiv.org/abs/1909.13447](https://arxiv.org/abs/1909.13447)

## Recommendation

Pick for benchmarking far-field multi-microphone ASR, speaker diarization, and source separation ("cocktail-party") in informal conversational English. The corpus is small (~5h total) — use as an evaluation set, not primary training data.

## Getting the data

Obtain from the [dataset homepage](https://www.amazon.science/publications/dipco-dinner-party-corpus).

Zenodo mirror at https://zenodo.org/records/8122551; verify array geometry expectations before plugging into a pipeline.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
