<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# MSP-Podcast

Large-scale naturalistic emotional speech mined from Creative-Commons podcasts, multi-rater annotated with categorical emotions and valence/ arousal/dominance attributes.

- **Task:** emotion, speaker
- **Languages:** en
- **Domain:** podcast (natural/spontaneous)
- **License:** see source terms
- **Homepage:** [https://www.lab-msp.com/MSP/MSP-Podcast.html](https://www.lab-msp.com/MSP/MSP-Podcast.html)
- **Paper:** [https://arxiv.org/abs/2509.09791](https://arxiv.org/abs/2509.09791)

## Recommendation

Best for realistic, in-the-wild speech emotion recognition at scale and the standard benchmark for recent SER challenges. Size is version-dependent and grows per release; natural emotion yields lower inter-rater agreement; access is gated behind a signed institutional agreement.

## Getting the data

Obtain from the [dataset homepage](https://www.lab-msp.com/MSP/MSP-Podcast.html).

Requires an institution-signed academic license (free); released by UT Dallas MSP Lab. Continually expanding, so size grows per release.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/emotion-recognize.yaml` — run it with `vkit docker run`.
