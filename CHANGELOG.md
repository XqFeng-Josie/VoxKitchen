# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- Four new ingest recipes (catalog count of recipe-backed datasets grows
  9 → 13): `libritts_r` (LibriTTS-R, en, CC BY 4.0), `hifitts` (Hi-Fi TTS,
  en multi-speaker, CC BY 4.0), `thchs30` (THCHS-30, zh, Apache-2.0),
  `thorsten_voice` (Thorsten-Voice 2021.02 neutral, de, CC0-1.0). All four
  are now downloadable via `vkit docker download <name> --root <dir>` and
  parseable via `vkit ingest --source recipe --recipe <name>`. The matching
  catalog entries flip from `manual` to `recipe-backed`; the docs index now
  shows "Access: recipe" with the OpenSLR size range.
- `vkit card <cuts.jsonl.gz>`: generate a standalone, shareable HTML dataset
  card (quality distributions, language/gender breakdowns, metrics summary, and
  sample utterances) from a processed CutSet. Uses the `viz` extra.
  Pass `--catalog-id <id>` to auto-fill title, description, and a Source section
  (license, homepage, paper, recommendation) from `voxkitchen/datasets/catalog.yaml`.
- Third-party operator plugins: install a package exposing operators via the
  `voxkitchen.operators` entry-point group and they are discovered, listed by
  `vkit operators` / `vkit doctor`, and usable in pipelines. Added
  `OPERATOR_API_VERSION`, a plugin authoring guide, and an example package
  (`examples/plugin-operator/`).
- Dataset catalog (`docs/datasets/`): a decision-support catalog of speech
  datasets generated from `voxkitchen/datasets/catalog.yaml`, with per-dataset
  recommendations, access links, and (where available) a download recipe and a
  suggested processing pipeline. Searchable via the docs site; browse by task.
  Now 60 entries — added (most recent batch) VoxCeleb1, TIMIT, MGB-2 (Arabic),
  SEAME (zh-en code-switching), Earnings-21, MyST (children's speech), AVSpeech
  (audio-visual), Opencpop (zh singing), DiPCo (far-field), SLUE (spoken
  language understanding); previously THCHS-30, MagicData-RAMC, Shrutilipi,
  AISHELL-2, JSUT, Thorsten-Voice, LibriTTS-R, MELD, MSP-IMPROV, IMDA NSC.
- Declarative field contracts (`reads`/`writes`/`optional_reads`/`clears`) on
  every operator, making data-flow dependencies machine-readable.
- Static pre-flight validation in `vkit validate` and as a fail-fast gate before
  every `vkit run` / `vkit docker run` (and in `--dry-run`): broken operator
  chains — a stage requiring a field no upstream stage produces, or a
  `quality_score_filter` condition referencing an absent metric — are reported
  before the executor starts, not mid-run. Pass `--no-preflight` to `vkit
  validate`, `vkit run`, or `vkit docker run` (forwarded to the in-container
  run) to skip the check.
- `normalize_text` operator: strips model-specific markup (e.g. SenseVoice
  emotion/language tags) and collapses whitespace (e.g. Paraformer
  inter-character spaces) in ASR transcripts before downstream use.
- `pack_jsonl` now exports `word_alignments` when present on a cut.
- Simplified Chinese README (`README.zh-CN.md`), a full translation of
  the top-level README. The English and Chinese READMEs cross-link each
  other in the header; all deeper docs remain English-only.
- Both READMEs now show a collapsed "Example output" block under Quick
  Start with the actual `vkit docker run` + `vkit inspect run` logs for
  the bundled `demo-no-asr` pipeline.
- `CITATION.cff` and a "Citation" section in both READMEs, so GitHub
  renders a "Cite this repository" entry for research use.
- `scripts/release.sh <version> --docker-only` skips the tag + git push
  steps and goes straight to the Docker build/push loop. Useful when
  PyPI publish already succeeded from an earlier run (or from a
  different machine) and you only need to (re)build the GHCR images.
  Mutually exclusive with `--replace-unpublished`. Pre-flight CHANGELOG
  and CI checks still run because the Docker build bakes the current
  source into the image.

### Changed

- Dropped the "Pre-alpha" status banner from the README and bumped the
  PyPI `Development Status` classifier from `2 - Pre-Alpha` to
  `4 - Beta` to match the current v0.3.x surface.
- `vkit run --dry-run` (and `vkit docker run --dry-run`) clarifies its
  GPU reporting. The `gpus:` header line is now labelled `(requested)`,
  and when any stage prefers the GPU executor the stage table is
  followed by a note that the Device column shows the *preferred*
  executor — `gpu` stages run on GPU when one is available and fall back
  to CPU otherwise. This removes the confusion where a `gpu` stage such
  as `silero_vad` appeared to require a GPU even though it runs fine on
  the CPU-only `slim` image.

### Fixed

- `tts-data-prep` template now normalizes text and performs forced alignment as
  its documentation claims; previously it declared word-level alignment but ran
  neither step. The ASR stage no longer performs a redundant internal alignment
  pass.
- Capped `numpy<2.3` in the `dev` optional-dependency group. `numba`
  (pulled in by `librosa` via the `segment` extra) supports `numpy<=2.2`,
  so a fresh `pip install -e .[dev]` on a host with a newer NumPy failed
  to import `librosa` and broke local test collection. The core
  `numpy>=1.24` runtime bound is unchanged.

## [0.3.0] — 2026-05-21

### Added

- Published the `voxkitchen` launcher to PyPI. Users install with
  `pipx install voxkitchen` (or `pip install voxkitchen`) instead of fetching
  a GitHub archive zip. A `publish` GitHub Actions workflow builds wheel +
  sdist on every `v*` tag and uploads via PyPI Trusted Publishing (OIDC, no
  stored API tokens). The workflow also exposes a manual `workflow_dispatch`
  for TestPyPI dry-runs.
- PyPI version badge in `README.md`.
- Pipeline YAML interpolation gained POSIX-style fallbacks. `${env:VAR:-foo}`
  uses `foo` when `VAR` is unset or empty; `${env:VAR:?msg}` raises a
  `PipelineLoadError` with `msg` when `VAR` is missing. The original
  `${env:VAR}` form still fails loudly on unset variables and remains the
  default for required tokens.
- `vkit operators` gained two new ways to navigate the 51-operator catalog:
  `vkit operators --category <cat>` shows a single section (audio, segment,
  augment, annotate, quality, synthesize, pack, noop) and
  `vkit operators search <keyword>` lists operators whose name or first-line
  docstring contains `<keyword>` (case-insensitive). Empty matches exit with
  code 1 so scripts can branch on no-result.
- `vkit download` and `vkit ingest --source recipe` now warn host users when
  invoked outside a managed runtime, mirroring the existing `vkit run`
  warning. The recipe-side dependencies (e.g. `datasets`) live in the Docker
  images, not in the PyPI launcher, so the previous silent failure mode is
  replaced with a pointer to `vkit docker download <recipe>`. `vkit ingest
  --source dir` and `--source manifest` stay quiet — they work on the host
  with only the lightweight launcher deps.
- `vkit schema export` writes a JSON Schema for `pipeline.yaml` files,
  derived from `PipelineSpec.model_json_schema()` plus the registered
  operators. A snapshot is committed at `docs/schemas/pipeline.schema.json`
  and served via raw.githubusercontent.com. `vkit init` now writes a
  `# yaml-language-server: $schema=…` directive at the top of every
  scaffolded `pipeline.yaml`, so VS Code, Neovim, and JetBrains users get
  autocompletion on operator names and inline validation of the spec
  structure out of the box. See `docs/reference/schema.md` for editor setup.
- TTS tutorials split into three focused pages so each can grow
  independently as new engines land. `tts-data-prep.md` renamed to
  `tts-training-data.md` (quality gate for raw recordings used to train
  a TTS model). The combined `tts-synthesis.md` was split into
  `tts-speaker.md` (built-in / seed-sampled voices — `tts_kokoro`,
  `tts_chattts`, `tts_cosyvoice` `sft` mode) and `tts-voice-cloning.md`
  (short-reference cloning — `tts_cosyvoice` `zero_shot` /
  `cross_lingual`, `tts_fish_speech`). Each tutorial carries its own
  capability matrix filtered to the relevant engines, a Quick Start in
  both Python and YAML, per-engine config snippets, and a decision flow.
  `mkdocs.yml`, `docs/index.md`, `README.md` "What You Can Build", and
  the skill operator notes were updated to match.
- Five new ingest recipes complete the "common dataset" coverage,
  bringing the total registered to 9:
  - `ljspeech` — single-speaker English TTS baseline (24 h, 13.1k
    utterances), downloaded from data.keithito.com. Prefers
    normalized text over raw; preserves the raw form only when
    normalization changed it.
  - `aishell3` — multi-speaker Mandarin TTS (218 speakers, ~85 h),
    downloaded from OpenSLR/93. Splits the interleaved character +
    pinyin `content.txt` into supervision text (chars) and
    `cut.custom["pinyin"]`; enriches gender from `spk-info.txt`.
  - `libritts` — multi-speaker English TTS derived from LibriSpeech
    (OpenSLR/60). Prefers `*.normalized.txt` over `*.original.txt`;
    enriches gender from `speakers.tsv`. Same seven-subset
    partitioning as LibriSpeech.
  - `cnceleb` — CN-Celeb 1, Chinese speaker recognition (~130k
    utterances, 1000 speakers, 11 genres), from OpenSLR/82.
    Empty-text Supervisions carry speaker / language tags. Subsets
    `data` / `dev` / `eval` follow the canonical splits; overlapping
    subsets deduplicate.
  - `musan` — MUSAN augmentation corpus (~10 GB of non-transcribed
    noise / music / speech), from OpenSLR/17. Closes the loop with
    the existing `noise_augment` operator. Subsets pick which of the
    three top-level categories to ingest; sub-categories are
    preserved as `cut.custom["musan_subcategory"]`.
- Every auto-downloadable recipe ships HEAD-probed compressed-size
  metadata in a new `download_sizes` dict on the Recipe class. The
  size is surfaced in three places: a new Size column in
  `vkit recipes`, a per-subset "downloading X GB" log line inside
  `Recipe.download()`, and the table in `docs/reference/recipes.md`.
  Multi-subset recipes (LibriSpeech / LibriTTS / AISHELL-1) show a
  range like `299 MB - 28.5 GB`. Manual / HuggingFace recipes render
  as a dash.

### Changed

- The TTS training-data tutorial (`tts-training-data.md`) now opens
  with an explicit "quality gate" framing and ends with a Quality
  Checklist summarizing the five thresholds (sample rate, duration,
  SNR, text present, alignment present).
- The Download column of `vkit recipes` now derives the source label
  from each recipe's URL host (keithito / openslr / huggingface /
  bare hostname) instead of hard-coding "openslr" for every recipe
  with `download_urls`.
- Docker image builds now reuse wheel and layer caches across runs.
  `docker/Dockerfile` mounts a BuildKit cache at `/root/.cache/uv` on
  every `RUN uv pip install`, so torch / transformers / funasr wheels
  download once instead of being refetched in each of the five venvs
  (`UV_NO_CACHE=1` removed). `scripts/release.sh` now builds through
  `docker buildx` against a dedicated `voxkitchen-builder` (auto-created
  with the docker-container driver), and pushes layer descriptors to
  `ghcr.io/xqfeng-josie/voxkitchen:buildcache-<target>` via
  `--cache-to type=registry,mode=max`. First build from a cold cache
  still takes 1-2 h; subsequent rebuilds where only application source
  changed land in single-digit minutes per target. New "Faster
  rebuilds" section in `docs/docker-build.md` walks through both cache
  layers, the manual `docker buildx` command to consume the public
  registry cache from a fork, expected timings per scenario, and which
  changes invalidate which layers.

### Fixed

- Wheel no longer ships duplicate copies of
  `voxkitchen/templates/pipelines/*.yaml`. The redundant
  `[tool.hatch.build.targets.wheel.force-include]` block was removed;
  hatchling already includes non-Python files inside the package
  directory.
- Untagged dev builds (e.g. the publish workflow's manual TestPyPI
  dispatch) now produce PEP 440-compliant versions like
  `0.2.1.dev5` instead of `0.2.1.dev5+g<sha>`. PyPI and TestPyPI both
  reject the latter form. Set via `local_scheme = "no-local-version"`
  in `[tool.hatch.version].raw-options`.
- Removed `[wenet]`, `[speaker]`, and `[tts-fish-speech]`
  optional-dependency groups from `pyproject.toml` because they
  declared `pkg @ git+...` direct references, which PyPI rejects on
  upload. The Docker images now install these from git inside their
  respective `RUN` lines (`fish-speech` joins `wenet`'s existing
  pattern in `docker/Dockerfile`). Operators that declare
  `required_extras = ["wenet"|"tts-fish-speech"]` still route
  correctly via `EXTRA_TO_ENV` in
  `voxkitchen/runtime/env_resolver.py`.
- `vkit run` now exits with code 1 (runtime failure) when a stage
  raises `StageFailedError`, matching the rest of the CLI's exit-code
  convention. Previously it returned code 2, which the codebase
  reserves for invocation errors (unknown flag, missing docker
  binary, unknown operator category).
- `vkit ingest` inline error messages now use the same `error:`
  prefix the rest of the CLI prints, instead of rendering the whole
  line in red without context.
- `vkit inspect cuts <missing>` no longer dumps a full Python
  traceback; it prints a one-line
  `error: manifest does not exist: …` and exits 1. Corrupt or empty
  manifests are reported the same way.
- `vkit inspect run|errors|trace <missing>` now exit with code 1 on
  a missing work directory or an unknown cut id. They previously
  printed an error message but returned exit 0, so shell scripts
  treated the failure as success.
- `docs/reference/tools-api.md` now documents the full
  `voxkitchen.tools` API surface. `compute_speaker_similarity` and
  `tokenize_audio` had shipped but were missing from the reference
  page — added their import line, usage section, and runtime-image
  hint to bring the doc in line with the code.
- README image and documentation links now use absolute
  `raw.githubusercontent.com` / `github.com` URLs. PyPI's README
  renderer leaves relative paths intact and resolves them against
  `https://pypi.org/project/voxkitchen/`, which 404s; the project
  page's logo and pipeline diagram were broken on the first publish.
- `voxkitchen.utils.download.download_file` is now atomic and
  retryable. The body streams into `<dest>.partial` and is renamed
  into place only on success, so an aborted transfer can no longer
  be mistaken for a complete download on the next call. Up to three
  attempts are made on transient errors (ConnectionResetError /
  OSError / …) with exponential backoff (2s, 4s). This came out of
  real OpenSLR mid-stream RSTs hit while end-to-end-verifying the
  new AISHELL-3 and LibriTTS recipes.
- `cnceleb` recipe rewritten to match the real corpus layout.
  Verified against the live 22 GB tarball, the previous
  implementation was wrong about `dev.lst` (it lists speaker IDs,
  not paths) and about `eval` (audio lives in separate
  `eval/enroll` and `eval/test` flat directories, not as pointers
  into `data/`). Counts now match the paper: 126,532 cuts / 997
  speakers in data, 107,953 cuts / 797 speakers in dev, 17,973 cuts
  / 200 speakers in eval.
- Local release/push checks now run the same fast lint, format,
  typecheck, and pytest gate as CI via `scripts/check-ci.sh`.

### Removed

- The `tedlium3` recipe is removed entirely. The canonical
  `openslr.org/resources/51/` mirror was de-listed by the project
  upstream — every probe returns 404 and `www.openslr.org/51/`
  reports "Resource not found". Without a working auto-download URL,
  the recipe was effectively manual-only, and shipping a registered
  recipe whose `vkit docker download` is a guaranteed no-op was a
  UX cost without a corresponding benefit. The STM-parsing and
  slice layout logic remain in git history (commit 15e6d19 and its
  subsequent corrections); reintroduce via a HuggingFace-streaming
  recipe (modelled on `fleurs`) when a real data path is available.

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
