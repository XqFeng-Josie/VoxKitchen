# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed

- Local release/push checks now run the same fast lint, format, typecheck, and
  pytest gate as CI via `scripts/check-ci.sh`.

## [0.2.0] — 2026-05-18

### Added

- `vkit docker download` for Docker-first dataset downloads, with `slim`
  as the default recipe runtime.
- Docker image recommendations in `vkit validate <yaml>` and
  `vkit run <yaml> --dry-run`, including copyable `pull` and `run`
  commands.
- Packaged pipeline templates in the wheel, so `vkit init --template ...`
  works outside a source checkout.
- Agent-neutral VoxKitchen skill under `skill/` for Claude, Codex, and other
  `SKILL.md`-compatible agents.

### Changed

- Reoriented the user path around lightweight host `vkit` plus prebuilt Docker
  runtimes. User docs, examples, tutorials, generated project READMEs, and
  operator references now recommend `vkit docker ...` instead of local pip
  extras for runtime execution.
- `vkit run` is treated as the current-environment/container entrypoint and
  warns when used directly on a host.
- GPU stages now show per-cut progress bars, so long-running ASR/diarization/TTS
  stages provide live feedback inside Docker runs.
- `vkit docker build` and `scripts/release.sh` now default Docker client
  scratch/config/cache paths to project-local `./.docker`, with
  `VKIT_DOCKER_WORK_DIR` as the override.
- Quick start now uses the smaller `slim` demo path.
- Operator and doctor hints now point to Docker runtime tags instead of pip
  installation commands.
- Legacy source-tree Docker helper scripts were removed in favor of the
  maintained `vkit docker ...` subcommands.

### Fixed

- `vkit docker run` now forwards pipeline execution flags such as `--dry-run`,
  `--resume-from`, `--stop-at`, `--num-workers`, `--work-dir`, and
  `--keep-intermediates`.
- Docker wrapper mount behavior is more predictable: pipeline runs manage
  `./work` and `./output`, dataset downloads manage `./data`, and doctor avoids
  unnecessary user data/work/output mounts.
- `vkit run --dry-run` can validate operators via exported schemas when the
  current environment cannot import the operator directly.
- Documentation paths are consistently `./data`, `./work`, and `./output`.
- `speaker_embed` now defaults to the SpeechBrain backend in official Docker
  images, avoiding the upstream WeSpeaker/s3prl warmup failure on Python 3.11.
- `tts_fish_speech` now targets the Fish-Speech S2 inference engine and is
  expected in both `fish-speech` and `latest` images.
- Official Docker envs now pin `setuptools<81` for `pyworld`, fixing
  `pitch_stats` runtime failures caused by the removal of `pkg_resources`.
- `pack_*` export operators now run as whole-CutSet stages instead of sharded
  workers, preventing shared output directories from being overwritten by the
  last worker.
- Non-shardable batch operators now fail atomically instead of falling back to
  per-cut retries that could leave partial side-effect outputs behind.
- `report.html` now links to the correct VoxKitchen GitHub repository.

### Known limitations

- `speaker_embed` with `method: wespeaker` is experimental and intended for
  custom environments. Official Docker images use `method: speechbrain`.
- `utmos_score`: the `utmos` submodule is missing from the current
  speechmos PyPI wheel. Operator registers but runtime fails.
- `:latest` image is ~123 GB — larger than optimal because each per-env
  stage re-copies core's model_cache. Use per-env tags unless you need
  cross-cluster pipelines.

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
