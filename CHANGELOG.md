# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Scene-based pipeline templates**: `vkit init --template tts|asr|cleaning|speaker`
- **Documentation site**: tutorials, operator reference, CLI reference, YAML spec
- **CONTRIBUTING.md** and community guidelines

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
