# Changelog

All notable changes to VoxKitchen are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
  structure out of the box. See `docs/reference/schema.md` for editor
  setup.

### Fixed

- Wheel no longer ships duplicate copies of `voxkitchen/templates/pipelines/*.yaml`.
  The redundant `[tool.hatch.build.targets.wheel.force-include]` block was
  removed; hatchling already includes non-Python files inside the package
  directory.
- Untagged dev builds (e.g. the publish workflow's manual TestPyPI dispatch)
  now produce PEP 440-compliant versions like `0.2.1.dev5` instead of
  `0.2.1.dev5+g<sha>`. PyPI and TestPyPI both reject the latter form. Set via
  `local_scheme = "no-local-version"` in `[tool.hatch.version].raw-options`.
- Removed `[wenet]`, `[speaker]`, and `[tts-fish-speech]` optional-dependency
  groups from `pyproject.toml` because they declared `pkg @ git+...` direct
  references, which PyPI rejects on upload. These packages were never reachable
  via `pip install voxkitchen[...]` for end users anyway; the Docker images
  now install them from git inside their respective `RUN` lines
  (`fish-speech` joins `wenet`'s existing pattern in `docker/Dockerfile`).
  Operators that declare `required_extras = ["wenet"|"tts-fish-speech"]` still
  route correctly via `EXTRA_TO_ENV` in `voxkitchen/runtime/env_resolver.py`.
- `vkit run` now exits with code 1 (runtime failure) when a stage raises
  `StageFailedError`, matching the rest of the CLI's exit-code convention.
  Previously it returned code 2, which the codebase reserves for invocation
  errors (unknown flag, missing docker binary, unknown operator category).
- `vkit ingest` inline error messages now use the same `error:` prefix the
  rest of the CLI prints, instead of rendering the whole line in red without
  context.
- `vkit inspect cuts <missing>` no longer dumps a full Python traceback; it
  prints a one-line `error: manifest does not exist: …` and exits 1.
  Corrupt or empty manifests are reported the same way.
- `vkit inspect run|errors|trace <missing>` now exit with code 1 on a missing
  work directory or an unknown cut id. They previously printed an error
  message but returned exit 0, so shell scripts treated the failure as
  success.
- `docs/reference/tools-api.md` now documents the full
  `voxkitchen.tools` API surface. `compute_speaker_similarity` and
  `tokenize_audio` had shipped but were missing from the reference
  page — added their import line, usage section, and runtime-image hint
  to bring the doc in line with the code.
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
