# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [0.1.0] — 2026-04-19

Initial public release.

### Features

- **51 operators** across 7 categories: audio, segment, augment,
  quality, annotate (ASR / diarize / TTS / forced-align / etc.), pack.
- **Declarative YAML pipelines**: ingest → stages → output, with
  resumable stage checkpoints, per-cut error tolerance, GC, and full
  operator provenance.
- **Multi-env Docker architecture**: one image, multiple isolated
  Python envs (core / asr / diarize / tts / fish-speech). Pipeline
  runner dispatches each stage to the env that can run its operator,
  communicating via the existing disk checkpoints. Sidesteps the
  pyannote-vs-funasr-vs-fish-speech dep conflicts that prevent a
  single-env "everything" build.
- **One CLI, two execution modes**: `vkit <cmd>` runs locally
  (pip install), `vkit docker <cmd>` runs in a container. Same flags,
  same YAML.
- **Six Docker tags**: `slim` (~3 GB, CPU), `asr`, `diarize`, `tts`,
  `fish-speech`, `latest` (all five envs, ~25 GB).
- **Python tools API**: `voxkitchen.tools.transcribe()`,
  `detect_speech()`, `estimate_snr()`, `extract_speaker_embedding()`,
  `enhance_speech()`, `align_words()`, `synthesize()` — for embedding
  VoxKitchen in other Python code.
- **Dataset recipes**: LibriSpeech, CommonVoice, AISHELL, FLEURS.
- **Templates**: `vkit init -t {tts,asr,cleaning,speaker}` scaffolds
  working starter pipelines.

### Known limitations

- `tts_fish_speech` operator targets fish-speech 1.x API; fish-speech 2.0
  reshuffled its Python entry points. The `:fish-speech` image builds
  and pre-downloads the model, but the operator is excluded from
  `EXPECTED_OPERATORS["fish-speech"]` until a rewrite lands.
- `speaker_embed` warmup fails on Python 3.11 due to an s3prl dataclass
  bug (mutable default). The operator registers but runtime fails.
- `utmos_score`: speechmos' `utmos` submodule is absent in the current
  PyPI wheel. Operator registers; runtime fails.

### License

Apache 2.0.
