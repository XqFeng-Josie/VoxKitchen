<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# AVSpeech

A large-scale audio-visual dataset of ~4700 hours of 3-10 second clips drawn from ~290k YouTube videos, each segment featuring a single visible speaker with clean speech, released for the "Looking to Listen at the Cocktail Party" speech-separation work.

- **Task:** asr, speaker
- **Languages:** multi
- **Hours:** 4700
- **Domain:** youtube audio-visual
- **License:** see source terms
- **Homepage:** [https://looking-to-listen.github.io/avspeech/](https://looking-to-listen.github.io/avspeech/)
- **Paper:** [https://arxiv.org/abs/1804.03619](https://arxiv.org/abs/1804.03619)

## Recommendation

Pick for audio-visual speech separation, speaker-conditioned source separation, and lip-sync / talking-face research where clean single-speaker reference segments are needed; useful as a pretraining source for AV speech models. Distributed as CSV segment lists referencing YouTube — expect link rot and YouTube ToS constraints.

## Getting the data

Obtain from the [dataset homepage](https://looking-to-listen.github.io/avspeech/).

No transcripts; users must download clips from YouTube themselves under Google research terms + YouTube ToS.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `voxkitchen/templates/pipelines/speaker-analysis.yaml` — run it with `vkit docker run`.
