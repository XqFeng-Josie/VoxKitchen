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
- **Six Docker tags**: `slim` (~13 GB, CPU), `asr`, `diarize`, `tts`,
  `fish-speech`, `latest` (all five envs, ~103 GB).
- **Python tools API**: `voxkitchen.tools.transcribe()`,
  `detect_speech()`, `estimate_snr()`, `extract_speaker_embedding()`,
  `enhance_speech()`, `align_words()`, `synthesize()` — for embedding
  VoxKitchen in other Python code.
- **Dataset recipes**: LibriSpeech, CommonVoice, AISHELL, FLEURS.
- **Templates**: `vkit init -t {tts,asr,cleaning,speaker}` scaffolds
  working starter pipelines.

### Known limitations

- `tts_fish_speech`: operator targets fish-speech 1.x API; fish-speech 2.0
  reshuffled the Python entry points. Image builds, model is cached,
  operator is parked pending rewrite.
- `speaker_embed`: warmup fails on Python 3.11 (s3prl dataclass
  mutable-default bug). Operator registers but runtime fails.
- `utmos_score`: the `utmos` submodule is missing from the current
  speechmos PyPI wheel. Operator registers but runtime fails.
- `:latest` image is ~103 GB — larger than optimal because each per-env
  stage re-copies core's model_cache. Use per-env tags unless you need
  cross-cluster pipelines.

### License

Apache 2.0.
