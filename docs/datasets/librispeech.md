<!-- AUTO-GENERATED from voxkitchen/datasets/catalog.yaml — do not edit; run python -m voxkitchen.datasets.catalog_gen -->

# LibriSpeech

Read English audiobooks; the standard English ASR benchmark.

- **Task:** asr
- **Languages:** en
- **Hours:** 960
- **Domain:** audiobook
- **License:** CC BY 4.0
- **Homepage:** [https://www.openslr.org/12](https://www.openslr.org/12)
- **Paper:** [https://www.danielpovey.com/files/2015_icassp_librispeech.pdf](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)

## Recommendation

The default starting point for English ASR — clean, well-segmented read speech with transcripts. Prototype on train-clean-100; use the full 960 h for production. Not representative of conversational or noisy audio.


## Getting the data

Downloadable via VoxKitchen (`librispeech`, source: openslr, size: 299 MB - 28.5 GB):

```bash
vkit docker download --tag slim librispeech --root ./data/librispeech
```

Subsets: dev-clean, dev-other, test-clean, test-other, train-clean-100, train-clean-360, train-other-500.

## Suggested processing

A recommended VoxKitchen pipeline ships in the repository at `examples/pipelines/librispeech-asr.yaml` — run it with `vkit docker run`.
