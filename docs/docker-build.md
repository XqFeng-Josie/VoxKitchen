# Building VoxKitchen Docker images

Most users should use `vkit docker pull --tag <tag>` to fetch the prebuilt
images from
`ghcr.io/xqfeng-josie/voxkitchen:{slim,asr,diarize,tts,fish-speech,latest}`.
Build locally only when:

- Air-gapped / internal registries that cannot reach GHCR
- Apple Silicon / non-`linux/amd64` hosts until we ship multi-arch images
- Custom CUDA versions (the published images pin CUDA 12.4)
- Adding private operators or baking in your own models
- Security audits that require a reproducible build from source

## Targets

The single [`docker/Dockerfile`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docker/Dockerfile) exposes six
BuildKit targets:

| Target | What it contains | Approx. size | GPU |
|--------|------------------|--------------|-----|
| `slim`        | `core` venv only                          | ~13 GB  | no  |
| `asr`         | core + `asr` venv (faster-whisper / funasr / qwen3 / forced align) | ~48 GB | yes |
| `diarize`     | core + `diarize` venv (pyannote only)     | ~32 GB  | yes |
| `tts`         | core + `tts` venv (kokoro / ChatTTS / CosyVoice) | ~44 GB | yes |
| `fish-speech` | core + `fish-speech` venv (torch 2.8 stack, S2 model cached) | ~57 GB  | yes |
| `latest`      | all 5 envs merged in one image            | ~123 GB | yes |

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

`vkit docker build <target>` is the preferred local wrapper. It runs the same
Dockerfile build and defaults Docker client scratch paths to `./.docker`:

- `DOCKER_CONFIG=./.docker/config`
- `TMPDIR=./.docker/tmp`
- `BUILDX_CONFIG=./.docker/buildx`
- `XDG_CACHE_HOME=./.docker/cache`

Override the base directory with `VKIT_DOCKER_WORK_DIR`:

```bash
VKIT_DOCKER_WORK_DIR=/data2/xiaoqinfeng/workdir/VoxKitchen/.docker \
    vkit docker build asr
```

The release script uses the same default before building and pushing images.
If `DOCKER_CONFIG` is project-local, log in to GHCR with the same config:

```bash
mkdir -p .docker/config
DOCKER_CONFIG="$PWD/.docker/config" docker login ghcr.io
```

Important: these variables move Docker CLI temp/config/cache files only. Image
layers and BuildKit layer cache are stored by the Docker daemon's `data-root`,
which is commonly `/var/lib/docker` and may still fill `/`. To move that data,
configure Docker daemon storage, for example:

```json
{
  "data-root": "/data2/xiaoqinfeng/docker-data"
}
```

Put that in `/etc/docker/daemon.json` and restart Docker. Use a path outside
the repository; daemon layer storage is much larger than `./.docker`.

Each build ends with a per-env `vkit doctor --expect <env>` smoke test. If any
expected operator fails to register in its env, the build fails loudly rather
than shipping an image that *looks* healthy but breaks at runtime.

## Faster rebuilds (BuildKit cache + GHCR layer cache)

A full cold build of all six targets takes ~1-2 hours and re-downloads
several GB of wheels per env (torch alone is ~2.5 GB per CUDA venv).
Two layers of cache make rebuilds — and builds from a fresh machine —
much faster.

### 1. uv wheel cache (always on, no setup)

`docker/Dockerfile` mounts a BuildKit cache at `/root/.cache/uv` on
every `RUN uv pip install`:

```dockerfile
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/<env>/bin/python ...
```

The cache is keyed by `id=uv-cache` and shared across all stages, so
torch 2.4.1+cu124 downloads **once** and the same wheel is reused by
`core` / `asr` / `diarize` / `tts`; `fish-speech` reuses its own torch
2.8 wheel on the next rebuild. The cache mount never lands in a final
image layer — it lives in BuildKit's local cache volume.

This is enabled automatically by Docker's default BuildKit. Both
`docker build` and `docker buildx build` honor it without flags.

To inspect or clear the local BuildKit cache:

```bash
docker buildx du                              # bytes on disk per cache type
docker buildx prune --filter type=exec.cachemount   # clear uv wheel cache
docker buildx prune                           # clear everything
```

### 2. GHCR registry layer cache (release script only)

The uv cache above lives on a single machine. `scripts/release.sh`
also publishes BuildKit's **layer cache** to GHCR so it survives
across machines and across `docker buildx prune`:

- After every successful target build, the script pushes layer
  descriptors to `ghcr.io/xqfeng-josie/voxkitchen:buildcache-<target>`
  via `--cache-to type=registry,mode=max`.
- On the next run — on the same machine or any other one logged into
  GHCR — `--cache-from` pulls those descriptors and BuildKit reuses
  unchanged layers without re-running the underlying `RUN` step.

The `buildcache-<target>` tags are **not runnable images**; they
carry only layer manifests. They are public, so anyone can pull from
them as a read-only mirror; pushing requires `write:packages` on the
GHCR repo.

Registry cache requires the `docker-container` BuildKit driver — the
default `docker` driver only supports `cache-to type=inline`. The
release script creates and reuses a builder named `voxkitchen-builder`
automatically on first run:

```bash
docker buildx create --name voxkitchen-builder --driver docker-container --use
```

If you want the same registry-cache behavior from a local `vkit docker
build` (or a direct `docker buildx build`), build with buildx
yourself. Read-only consumption of the public cache needs no
authentication:

```bash
docker buildx create --name vk-builder --driver docker-container --use   # one-time

docker buildx build \
    --target asr \
    --cache-from type=registry,ref=ghcr.io/xqfeng-josie/voxkitchen:buildcache-asr \
    --load \
    -f docker/Dockerfile -t voxkitchen:asr .
```

To also *write* to the cache (for example, you maintain a private
fork and want your CI builds to share state), point `--cache-to` at a
repo you have write access to:

```bash
--cache-to type=registry,ref=ghcr.io/<your-user>/voxkitchen-cache:asr,mode=max
```

`mode=max` is important — `mode=min` only exports the final stage's
layer, which discards most of the savings for multi-stage Dockerfiles
like this one.

### Expected timings (release.sh, single machine)

| Scenario | `latest` |
|---|---|
| First build, cold cache | ~1-2 h |
| Rebuild after `voxkitchen/**` source change only | ~5-10 min |
| Rebuild on a fresh machine after GHCR cache pull | ~10-20 min (cache download dominates) |
| Rebuild after `pyproject.toml` extras change | ~30-60 min (re-runs the affected env's install) |

The single-env targets (`slim`, `asr`, `diarize`, `tts`, `fish-speech`)
each take a fraction of `latest`'s time because they build only one
env branch. Use `--target slim` for the fastest smoke test loop while
hacking on `core` operators.

### When cache misses happen

Cache lookup is keyed by the exact `RUN` command string plus all
preceding layer hashes. Common things that invalidate downstream
layers:

| Change | Invalidates from |
|---|---|
| `pyproject.toml` (any line) | `COPY pyproject.toml` → all later venv installs |
| `docker/constraints/<env>.txt` | that env's `RUN uv pip install` and everything after in the stage |
| `docker/Dockerfile` itself (e.g. add a `RUN`) | the changed line and everything after in the same stage |
| `voxkitchen/**` source | `COPY . .` (line ~124) and the final `pip install -e . --no-deps`; warmup + schema dump rerun |
| Base image tag (`pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`) | everything |

Warmup (model downloads) reruns whenever its `RUN` step's cache is
invalidated, which currently includes any source change. Treat
warmup as the most expensive step after wheel installs and re-order
Dockerfile edits accordingly if you find yourself paying for it often.

## Including the pyannote diarization model at build time

`pyannote/speaker-diarization-3.1` is gated. Two approaches:

**(a)** Supply the token during build — the model is baked into the image:

```bash
docker build --target latest --build-arg HF_TOKEN=hf_xxx \
    -f docker/Dockerfile -t voxkitchen:latest .
```

**(b)** Omit at build — pass `HF_TOKEN` at run time, pyannote downloads on
first use:

```bash
# Put HF_TOKEN=hf_xxx in ./.env, then run through the vkit wrapper.
vkit docker run --image voxkitchen:latest pipeline.yaml
```

You must accept the
[model agreement](https://huggingface.co/pyannote/speaker-diarization-3.1)
on HuggingFace with the same account as the token, or pyannote returns 403.

**Do not** bake a personal HF token into an image you then publish — it
remains retrievable in the image layer history.

## Checking what's inside an image

```bash
vkit docker doctor --image voxkitchen:latest          # per-env table
vkit docker doctor --image voxkitchen:latest --json   # machine-readable
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
   [`voxkitchen/runtime/env_resolver.py`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/voxkitchen/runtime/env_resolver.py)
   — this is the source of truth for "which env runs ops with this extra".
3. Add the extras name to the `pip install -e ".[...]"` line in the matching
   stage of `docker/Dockerfile` (core-env / asr-env / tts-env).
4. Add the operators it enables to `EXPECTED_OPERATORS` in
   [`voxkitchen/cli/doctor.py`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/voxkitchen/cli/doctor.py) under the same
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

- Publishing to GHCR — that lives in [`RELEASING.md`](https://github.com/XqFeng-Josie/VoxKitchen/blob/main/RELEASING.md)
  and is driven by `scripts/release.sh`. The release script is also
  what wires up the GHCR layer cache described above.
- Multi-arch builds (arm64). The buildx command is straightforward; the
  longer story is validating every extras group on arm64 wheels, which we
  haven't done.
