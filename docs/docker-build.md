# Building VoxKitchen Docker images

Most users should `docker pull` the pre-built images from
`ghcr.io/xqfeng-josie/voxkitchen:{slim,asr,diarize,tts,fish-speech,latest}`
(or use `vkit docker pull`). Build locally when:

- Air-gapped / internal registries that cannot reach GHCR
- Apple Silicon / non-`linux/amd64` hosts until we ship multi-arch images
- Custom CUDA versions (the published images pin CUDA 12.4)
- Adding private operators or baking in your own models
- Security audits that require a reproducible build from source

## Targets

The single [`docker/Dockerfile`](../docker/Dockerfile) exposes six
BuildKit targets:

| Target | What it contains | Base | Size | GPU |
|--------|------------------|------|------|-----|
| `slim`        | `core` venv only                          | ~3 GB  | no  |
| `asr`         | core + `asr` venv (faster-whisper / funasr / qwen3 / forced align) | ~10 GB | yes |
| `diarize`     | core + `diarize` venv (pyannote only)     | ~5 GB  | yes |
| `tts`         | core + `tts` venv (kokoro / ChatTTS / CosyVoice) | ~10 GB | yes |
| `fish-speech` | core + `fish-speech` venv (torch 2.8 stack) | ~6 GB  | yes |
| `latest`      | all 5 envs merged in one image            | ~25 GB | yes |

Base image for GPU targets: `pytorch/pytorch:2.4.1-cuda12.4`. Slim uses
`torch==2.4.1+cpu` inside the same base image; the CUDA libs are unused
and can be ignored on CPU-only hosts.

The five venvs are isolated:

- `core` / `asr` / `diarize` / `tts` share a torch 2.4 + numpy 1.x stack.
- `fish-speech` has its own torch 2.8 + numpy 2.1 stack (upstream requirement).
- Diarize is its own env (not bundled into `asr`) so the `:diarize` image
  can ship without the full ASR stack for users who only need speaker
  segmentation.
- Pipelines with a `tts_fish_speech` stage get that stage dispatched to the
  `fish-speech` env via subprocess; other stages stay wherever they belong.
- `:latest` is assembled via `COPY --from=<env>` so BuildKit can build
  each env branch in parallel.

Both use the PyTorch CUDA base image. The `slim` target installs torch-cpu
wheels into its `core` venv and is runnable without a GPU; the base image's
CUDA libraries are unused there.

## Why one image with multiple envs?

A single Python environment cannot host all 51 operators — `pyannote.audio>=4.0`
wants `torch>=2.8 + numpy>=2.1`, `funasr + modelscope` are capped at the
`transformers<5 / numpy<2` stack, and the four TTS engines each pull transitive
deps that fight the ASR stack on `transformers` / `ctranslate2` versions.

VoxKitchen already checkpoints every pipeline stage to `cuts.jsonl.gz` on disk.
Stages don't share in-memory state, so they can run in different Python
interpreters if the parent runner routes them. The `latest` image ships five
venvs at `/opt/voxkitchen/envs/{core,asr,diarize,tts,fish-speech}/` and a
mapping from operator name to env at `/opt/voxkitchen/op_env_map.json`. The
pipeline runner in the `core` venv reads that map and spawns `stage_runner`
in the target venv for cross-env stages. Stages communicate only via the
existing on-disk checkpoints.

Full design: [`docs/architecture/multi-env.md`](architecture/multi-env.md).

## Build commands

```bash
docker build --target slim   -f docker/Dockerfile -t voxkitchen:slim   .
docker build --target latest -f docker/Dockerfile -t voxkitchen:latest .
```

Each build ends with a per-env `vkit doctor --expect <env>` smoke test. If any
expected operator fails to register in its env, the build fails loudly rather
than shipping an image that *looks* healthy but breaks at runtime.

### Including the pyannote diarization model at build time

`pyannote/speaker-diarization-3.1` is gated. Two approaches:

**(a)** Supply the token during build — the model is baked into the image:

```bash
docker build --target latest --build-arg HF_TOKEN=hf_xxx \
    -f docker/Dockerfile -t voxkitchen:latest .
```

**(b)** Omit at build — pass at run time, pyannote downloads on first use:

```bash
docker run --rm --gpus all -e HF_TOKEN=hf_xxx -v /data:/data \
    voxkitchen:latest run pipeline.yaml
```

You must accept the
[model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1)
on HuggingFace with the same account as the token, or pyannote returns 403.

**Do not** bake a personal HF token into an image you then publish — it
remains retrievable in the image layer history.

## Checking what's inside an image

```bash
docker run --rm voxkitchen:latest doctor          # per-env table
docker run --rm voxkitchen:latest doctor --json   # machine-readable
```

`doctor` in the `latest` image auto-detects the five envs and runs each
one's self-check in a subprocess, producing one row per env plus a detail
list of any missing operators. In `slim` (or any single-env target) it
shows the single-env report.

During image build we also write `/opt/voxkitchen/warmup_<env>.json` for each
env; `doctor` reads these to report model-cache status.

## CUDA version mismatch

Both targets pin CUDA 12.4 via the `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`
base image. If your driver only supports an older CUDA, edit the first `FROM`
line in `docker/Dockerfile` to match — PyTorch publishes matching base images
at `pytorch/pytorch:2.4.1-cuda{11.8,12.1}-cudnn9-runtime`. You will also need
to change the `--index-url https://download.pytorch.org/whl/cu124` lines in
the asr-env and tts-env stages to the matching CUDA version.

Keep `torch==2.4.1` in `docker/constraints/*.txt` unless you intentionally want
a different torch minor — some extras pin specific torch versions and relaxing
this here will not help if they re-pin downstream.

## Adding a new extras group

1. Add the group to `pyproject.toml` under `[project.optional-dependencies]`.
2. Decide which env should ship it. Update `EXTRA_TO_ENV` in
   [`voxkitchen/runtime/env_resolver.py`](../voxkitchen/runtime/env_resolver.py)
   — this is the source of truth for "which env runs ops with this extra".
3. Add the extras name to the `pip install -e ".[...]"` line in the matching
   stage of `docker/Dockerfile` (core-env / asr-env / tts-env).
4. Add the operators it enables to `EXPECTED_OPERATORS` in
   [`voxkitchen/cli/doctor.py`](../voxkitchen/cli/doctor.py) under the same
   env key, so the build-time smoke test catches regressions.
5. If the new extras introduces a shared dep that conflicts with existing
   pins, update `docker/constraints/{core,asr,tts}.txt` to a mutually
   compatible version — or, if no compatible version exists, the new extras
   belongs in a new env (expand the design, don't force-merge).

## The `gender` extras gotcha

`inaSpeechSegmenter` pins `tensorflow[and-cuda]` + `onnxruntime-gpu` which
won't install alongside either the asr or the tts stack. The `gender` extras
group is therefore not installed in any image. The `gender_classify` operator
is still registered and works with `method=f0` (pitch-based, no model) or
`method=speechbrain` (uses the `classify` extras, which IS installed in core).
Users who truly need `method=inaspeechsegmenter` can install it at runtime in
a throwaway container: `pip install inaSpeechSegmenter` inside the running
image and run again.

## Debugging a failed build

1. `docker build --target slim` first — it's smaller and catches basic apt /
   pyproject / uv issues before you spend time on GPU layers.
2. If slim passes but `latest` fails in asr-env or tts-env, the issue is
   almost always a transitive dep conflict in that env. Re-run that stage
   manually:

   ```bash
   docker build --target core-env -f docker/Dockerfile -t vk:debug .
   docker run --rm -it vk:debug bash
   # inside:
   uv pip install --python /opt/voxkitchen/envs/core/bin/python -e ".[asr,funasr]"
   ```

3. `warmup_*.json` being empty / missing in a built image means warmup
   failed silently (we use `|| true` so missing models don't fail the build).
   `vkit doctor` surfaces which models didn't download.

## What this doc doesn't cover

- Publishing to GHCR — the CI workflow for that will land later, kept separate
  from the Dockerfile itself so local builds don't depend on secrets / auth.
- Multi-arch builds (arm64). The buildx command is straightforward; the
  longer story is validating every extras group on arm64 wheels, which we
  haven't done.
