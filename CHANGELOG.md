# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

#### Multi-env Docker architecture
One image, multiple isolated Python envs. Stages dispatch across envs via
subprocess; stages communicate through the existing disk checkpoints so no
in-memory state crosses the env boundary. Resolves the long-standing
"can't install every operator in one Python env" problem (pyannote 4 vs
funasr vs ChatTTS vs fish-speech each want incompatible torch/numpy pins).

- Six Docker tags, one Dockerfile:
  | Tag | Envs | Size |
  |---|---|---|
  | `voxkitchen:slim` | core | ~3 GB, CPU |
  | `voxkitchen:asr` | core + asr | ~10 GB |
  | `voxkitchen:diarize` | core + diarize (pyannote only) | ~5 GB |
  | `voxkitchen:tts` | core + tts (kokoro/ChatTTS/CosyVoice) | ~10 GB |
  | `voxkitchen:fish-speech` | core + fish-speech (isolated torch 2.8) | ~6 GB |
  | `voxkitchen:latest` | everything, 5 envs merged | ~25 GB |
- New runtime package: `voxkitchen.runtime` (env_resolver, stage_runner,
  dispatch, schemas, dump_schemas, merge_schemas).
- New `vkit doctor` command — aggregates per-env operator availability
  and model-cache status across all envs in the image.
- `vkit validate` now falls back to JSON-schema validation against
  `op_schemas.json` when an operator's class is not importable in the
  parent env (multi-env image parent env cannot import every operator).
- Helper scripts: `scripts/vkit-build.sh` (reads `.env`, picks target),
  `scripts/vkit-docker.sh` (pins `--user`, auto-mounts `./work` and
  `./data`, auto-loads `.env`, GPU autodetection).
- Pre-downloaded models land under `/opt/voxkitchen/model_cache/` (world-
  readable) so non-root runtime users can read every cached weight; the
  cache subtree is world-writable so runtime downloads land there too.

#### Other
- **Scene-based pipeline templates**: `vkit init --template tts|asr|cleaning|speaker`
- **Documentation site**: tutorials, operator reference, CLI reference, YAML spec
- **CONTRIBUTING.md** and community guidelines

### Changed
- `origin_start` / `origin_end` fields in `pack_jsonl` / `pack_manifest`
  now correctly report each cut's position in the **original source file**
  (previously stuck at the first materializing stage's offsets, so every
  VAD segment inherited `0, recording_duration` — wrong). Fix chains
  offsets through ffmpeg_convert / resample.
- `dnsmos` extras now declares `onnxruntime>=1.17,<2` explicitly —
  `speechmos` itself doesn't declare it, so `pip install voxkitchen[dnsmos]`
  used to fail at runtime.
- `numpy` core dependency bound relaxed from `<2.0` to unbounded; the
  fish-speech env needs numpy 2.x, other envs stay on 1.x via their
  per-env constraints files under `docker/constraints/`.
- The 3-image interim Docker layout (`Dockerfile` + `Dockerfile.cpu`,
  per-image `docker/Dockerfile.{core,asr,tts}`) is retired in favor of
  the single multi-env `docker/Dockerfile`.

### Known limitations
- `tts_fish_speech` operator still targets fish-speech 1.x API; upstream
  reshuffled to `fish_speech.inference_engine.TTSInferenceEngine` in 2.0.
  The `:fish-speech` env builds and the model pre-downloads, but the
  operator is excluded from `EXPECTED_OPERATORS["fish-speech"]` until
  a follow-up PR rewrites it against the 2.0 API.
- `speaker_embed` warmup fails at build time against Python 3.11 due to
  an s3prl upstream dataclass bug (mutable default). The operator is
  registered but will fail at runtime until s3prl ships a fix.
- `utmos_score` warmup skips — `speechmos.utmos` submodule is missing in
  the current pypi wheel. Operator still registers.

## [0.2.0] — 2026-04-15

### Added

#### New Operators (+9)
- `speed_perturb` — speed perturbation via resampling
- `volume_perturb` — random volume gain in dB range
- `noise_augment` — mix with noise files at random SNR
- `reverb_augment` — convolve with Room Impulse Responses
- `speaker_embed` — speaker embedding extraction (WeSpeaker / SpeechBrain)
- `speech_enhance` — neural denoising via DeepFilterNet
- `forced_align` — word-level forced alignment via Qwen3-ForcedAligner
- `emotion_recognize` — speech emotion detection via emotion2vec
- `qwen3_asr` — Qwen3-ASR (30 languages + 22 Chinese dialects)

#### Infrastructure
- **Lazy CutSet**: `CutSet.from_jsonl_gz(path, lazy=True)` for large manifests
- **Stage statistics**: `_stats.json` with wall time and throughput per stage
- **`vkit download`**: download datasets (`librispeech`, `aishell`, `fleurs`)
- **FLEURS recipe**: Google's 102-language dataset via HuggingFace
- **Operator categories**: `vkit operators` groups by Audio/Segmentation/Augmentation/etc.
- **Class docstrings**: all 43 operators show descriptions in CLI
- **Python tools API**: `extract_speaker_embedding()`, `enhance_speech()`, `align_words()`

#### Download Support
- LibriSpeech: all 7 subsets from openslr.org
- AISHELL-1: data + resource from openslr.org
- FLEURS: 102 languages via HuggingFace `datasets`

### Changed
- `vkit operators` now shows category grouping and usage hint
- `vkit inspect run` now shows wall time and throughput per stage
- `forced_align` switched from ctc-forced-aligner (MMS_FA) to Qwen3-ForcedAligner

### Fixed
- `wespeaker` expects torch tensor, not numpy array
- `speech_enhance` API: `init_df()` not `init_df_model()`
- `forced_align` text normalization for better alignment quality

## [0.1.0] — 2026-04-12

### Added
- Initial release with 34 operators
- Declarative YAML pipeline engine with resume, error tolerance, GC
- 3 recipes: LibriSpeech, CommonVoice, AISHELL
- CLI: `vkit run`, `vkit validate`, `vkit inspect`, `vkit operators`, `vkit viz`
- Python tools API: `transcribe()`, `detect_speech()`, `estimate_snr()`, etc.
- HTML report generation
- Gradio interactive panel
