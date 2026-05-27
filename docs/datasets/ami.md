<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AMI Meeting Corpus

~100 h of recorded English meetings with synchronized audio, video, and rich annotations including transcripts and speaker labels.

- **Task:** asr, speaker
- **Languages:** en
- **Hours:** 100
- **Domain:** meetings
- **License:** CC BY 4.0
- **Homepage:** [https://groups.inf.ed.ac.uk/ami/corpus/](https://groups.inf.ed.ac.uk/ami/corpus/)
- **Paper:** [https://dl.acm.org/doi/10.1007/11677482_3](https://dl.acm.org/doi/10.1007/11677482_3)

## Recommendation

A standard benchmark for meeting-domain ASR, speaker diarization, and overlapping speech, with headset and far-field mic conditions. Choose it when you need multi-party conversational audio. Only ~100 h and largely non-native English, so it is small for training from scratch.

## Getting the data

Obtain from the [dataset homepage](https://groups.inf.ed.ac.uk/ami/corpus/).

Also accessible via OpenSLR (openslr.org/16) and the HF mirror edinburghcstr/ami.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
