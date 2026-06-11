# syntax=docker/dockerfile:1.6
# ============================================================
# VoxKitchen: one Dockerfile, six build targets, five isolated envs.
#
# Targets (use --target <name>):
#   slim          core env only               ~13 GB   CPU
#   asr           core + asr                  ~48 GB   GPU
#   diarize       core + pyannote             ~32 GB   GPU
#   tts           core + kokoro/chattts/cv    ~44 GB   GPU
#   fish-speech   core + fish-speech (t2.8)   ~57 GB   GPU
#   latest        all five envs merged        ~123 GB  GPU
#
# Examples:
#   docker build --target slim    -t voxkitchen:slim    -f docker/Dockerfile .
#   docker build --target latest  -t voxkitchen:latest  -f docker/Dockerfile .
#   docker build --target latest --build-arg HF_TOKEN=hf_xxx \
#       -t voxkitchen:latest -f docker/Dockerfile .    # bakes pyannote in
#
# Or via the vkit CLI wrapper (reads HF_TOKEN from ./.env automatically):
#   vkit docker build latest
#
# See docs/architecture/multi-env.md for the design.
# ============================================================

# Global build args (available in every stage that re-declares them).
ARG HF_TOKEN=""
ARG VOXKITCHEN_VERSION="0.0.0.dev0"

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS base

ARG VOXKITCHEN_VERSION

# uv is ~10x faster than pip and handles per-venv deps cleanly.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    # Scope the pretend version to voxkitchen only. Release builds pass
    # VOXKITCHEN_VERSION; local Docker builds keep the dev fallback. A bare
    # SETUPTOOLS_SCM_PRETEND_VERSION leaks into every setuptools-scm
    # build in the venv — observed breaking pyopenjtalk (misaki[ja] dep)
    # because its wheel then declared 0.0.0.dev0 != the requested 0.4.1.
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXKITCHEN=${VOXKITCHEN_VERSION} \
    # uv writes wheels here; every `uv pip install` below mounts a BuildKit
    # cache at this path (id=uv-cache, shared across stages). This lets the
    # second venv that wants torch==2.4.1+cu124 — and every subsequent build
    # — reuse the wheels already downloaded by the first one. The cache
    # itself is not part of the final image layer.
    UV_CACHE_DIR=/root/.cache/uv \
    # Shared model-cache root. Default locations ($HOME/.cache/...) are
    # root-owned mode 700 during build and unreachable when users run
    # with --user $(id -u):$(id -g). Re-point every model library we use
    # to a single path under /opt/voxkitchen/ so pre-downloaded models
    # stay visible regardless of container UID.
    HF_HOME=/opt/voxkitchen/model_cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/voxkitchen/model_cache/huggingface/hub \
    TRANSFORMERS_CACHE=/opt/voxkitchen/model_cache/huggingface \
    TORCH_HOME=/opt/voxkitchen/model_cache/torch \
    MODELSCOPE_CACHE=/opt/voxkitchen/model_cache/modelscope \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    # HF's new Xet storage backend flakes out on pyannote repo (observed:
    # 416 Range Not Satisfiable against cas-server.xethub.hf.co). Fall
    # back to the stable LFS path for all HF downloads. Saves us from
    # chasing xet-core bugs that are out of our hands.
    HF_HUB_DISABLE_XET=1

# System deps used across envs:
#   build-essential + python3-dev — for C extensions shipped as sdist
#     (webrtcvad, pyworld, pyaudio).
#   portaudio19-dev — pyaudio (transitive from fish-speech) needs portaudio.h
#     to build and libportaudio2 at runtime.
#   ffmpeg + libsndfile1 + sox — audio I/O.
#   espeak-ng — phonemizer backend for kokoro/fish-speech.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        pkg-config \
        build-essential \
        python3-dev \
        portaudio19-dev \
        ffmpeg \
        libsndfile1 \
        espeak-ng \
        sox \
    && mkdir -p /tmp/numba-cache \
    && chmod 1777 /tmp/numba-cache \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject + a stub package first so pip install is cacheable even
# when source files change. Real source is copied at the end of each env
# stage and reinstalled with --no-deps.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p voxkitchen && \
    mkdir -p voxkitchen/templates/pipelines && \
    echo '__version__ = "0.0.0.dev0"' > voxkitchen/_version.py && \
    echo '' > voxkitchen/__init__.py

COPY docker/constraints/ /app/docker/constraints/

# ============================================================
# core-env: CPU torch, lightweight deps. Parent env of any build.
# ============================================================
FROM base AS core-env

RUN uv venv /opt/voxkitchen/envs/core --python 3.11

# CPU PyTorch from the dedicated index (uv respects --index-url per-package).
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

# Core package (no extras yet). MUST succeed.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        -e .

# Core extras. [gender] is intentionally excluded everywhere — inaSpeechSegmenter
# pulls tensorflow[and-cuda] which fights every other dep cluster. The
# gender_classify operator still registers and works via method=f0 / method=speechbrain.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,viz,viz-panel]"

# Full source (overwrites the stub). Reinstall with --no-deps so the resolver
# doesn't touch already-installed extras.
COPY . .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/core/bin/python -e . --no-deps

# Pre-download core models; dump this env's operator schemas for later merge.
RUN /opt/voxkitchen/envs/core/bin/python scripts/warmup_models.py --group core || true
RUN mkdir -p /opt/voxkitchen && \
    /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.dump_schemas \
        --env core --out /opt/voxkitchen/schemas_core.json

# ============================================================
# slim (target): core env only, lightweight image
# ============================================================
FROM core-env AS slim

# Merge the (single) schema dump into op_schemas.json + op_env_map.json.
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json

# Build-time smoke test. Fails the build if any expected core operator
# didn't register (i.e. an extras group silently failed to install).
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core

# World-readable for non-root runtime; see latest-target comment.
RUN chmod -R a+rX /opt/voxkitchen

LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen slim" \
      org.opencontainers.image.description="core env only — VAD, quality, packing, speaker embed, enhancement (CPU)"

ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]

# ============================================================
# asr-env: layered on core-env
# ============================================================
FROM core-env AS asr-env

# Re-import the global ARG so this stage can see it (Docker ARG scoping).
ARG HF_TOKEN

RUN uv venv /opt/voxkitchen/envs/asr --python 3.11

# CUDA 12.4 torch first — pinned via constraint so the later extras install
# can't re-resolve it. PyPI default torch ships CUDA 12.1 wheels which
# work on a 12.4 driver but we want version alignment.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

# ASR env inherits everything core has + ASR-specific extras. diarize
# is deliberately NOT here — it lives in its own env so a ":diarize"
# image can ship without the full ASR stack.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,viz,viz-panel,asr,whisper,funasr,align]"

# wenet has no PyPI release, and its repo's .gitmodules references a submodule
# (runtime/gpu/.../FasterTransformer) with no URL, so uv's automatic
# `git submodule update --init` for a `git+https` dep fails. We only need the
# Python package (not the GPU C++ runtime), so clone WITHOUT submodules and
# install from the local checkout, pinned for reproducibility. The asr.txt
# constraint keeps torch/torchaudio at 2.4.1 — wenet imports cleanly there;
# newer torch dropped the private `Union` re-export wenet relies on.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    git clone --filter=blob:none https://github.com/wenet-e2e/wenet.git /opt/wenet-src && \
    git -C /opt/wenet-src checkout 51b577282dd62a84596fce381aab930f9f62b547 && \
    uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt /opt/wenet-src

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/asr/bin/python -e . --no-deps

RUN HF_TOKEN=${HF_TOKEN} /opt/voxkitchen/envs/asr/bin/python scripts/warmup_models.py --group asr || true
RUN /opt/voxkitchen/envs/asr/bin/python -m voxkitchen.runtime.dump_schemas \
        --env asr --out /opt/voxkitchen/schemas_asr.json

# ============================================================
# diarize-env: just pyannote + core-extras. Much smaller than asr-env
# because no funasr / whisper / qwen3. Exists so the `:diarize` target
# stays focused for users who only need speaker diarization.
# ============================================================
FROM core-env AS diarize-env

ARG HF_TOKEN

RUN uv venv /opt/voxkitchen/envs/diarize --python 3.11

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/diarize/bin/python \
        -c /app/docker/constraints/diarize.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/diarize/bin/python \
        -c /app/docker/constraints/diarize.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,viz,viz-panel,diarize]"

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/diarize/bin/python -e . --no-deps

RUN HF_TOKEN=${HF_TOKEN} /opt/voxkitchen/envs/diarize/bin/python scripts/warmup_models.py --group diarize || true
RUN /opt/voxkitchen/envs/diarize/bin/python -m voxkitchen.runtime.dump_schemas \
        --env diarize --out /opt/voxkitchen/schemas_diarize.json

# ============================================================
# tts-env: layered on core-env (independent of asr)
# ============================================================
FROM core-env AS tts-env

RUN uv venv /opt/voxkitchen/envs/tts --python 3.11

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,viz,viz-panel,tts-kokoro,tts-chattts,tts-cosyvoice]"

# misaki (pulled by kokoro) lazy-downloads the spacy English model at first
# TTS call via `uv pip install`. In a non-root runtime that lazy install
# fails (can't write to /opt/voxkitchen/envs/*). Install it now so every
# runtime call is a cache hit.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# CosyVoice is not on PyPI and its repo has no setup.py/pyproject.toml, so
# `pip install git+https://...` is not possible. Instead we clone the repo at
# a pinned SHA directly into /opt/cosyvoice and register it on the tts venv's
# sys.path via a .pth file. The Matcha-TTS submodule is satisfied by the PyPI
# `matcha-tts` package (same BASECFM API). Pinned to:
#   FunAudioLLM/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
# (latest as of 2026-06-04; re-confirmed AutoModel export present).
RUN git clone --depth=1 \
        https://github.com/FunAudioLLM/CosyVoice.git \
        /opt/cosyvoice && \
    cd /opt/cosyvoice && \
    git fetch --depth=1 origin 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc && \
    git checkout 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc

# Install the pip-installable deps that CosyVoice needs but are not yet
# present in the tts env (torch, torchaudio, numpy, modelscope, onnxruntime
# are already installed above via the tts-cosyvoice / dnsmos extras).
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt \
        "HyperPyYAML>=1.2" \
        "matcha-tts==0.0.7.0" \
        "inflect>=7.0" \
        "omegaconf>=2.3" \
        "openai-whisper>=20231117"

# matcha-tts (above) pulls in upstream `phonemizer`, which shares the
# `phonemizer/` import namespace with `phonemizer-fork` (a misaki/kokoro
# dependency) and shadows it. Upstream's EspeakWrapper lacks `set_data_path`,
# which misaki 0.9.x calls at import — so kokoro dies with AttributeError.
# Remove the upstream package and reinstall the fork so its files own the
# `phonemizer/` namespace. Must run AFTER the matcha-tts install above and
# BEFORE the tts warmup below (which exercises kokoro). Verified: fork alone
# → EspeakWrapper.set_data_path present → kokoro synthesizes.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip uninstall --python /opt/voxkitchen/envs/tts/bin/python phonemizer && \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt --reinstall phonemizer-fork==3.3.2

# Register /opt/cosyvoice on the tts venv's sys.path via a .pth file.
# This is the standard Python path-extension mechanism and survives across
# subprocess launches (PYTHONPATH would only apply to the current process).
RUN TTS_SITE=$(/opt/voxkitchen/envs/tts/bin/python \
        -c "import site; print(site.getsitepackages()[0])") && \
    echo "/opt/cosyvoice" > "${TTS_SITE}/cosyvoice.pth"

# Fish-Speech is isolated in fish-speech-env below because upstream pins a
# torch 2.8 + numpy 2.x stack. Keep this tts env focused on Kokoro, ChatTTS,
# and CosyVoice so those engines stay on the validated torch 2.4 stack.

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/tts/bin/python -e . --no-deps

RUN /opt/voxkitchen/envs/tts/bin/python scripts/warmup_models.py --group tts || true
RUN /opt/voxkitchen/envs/tts/bin/python -m voxkitchen.runtime.dump_schemas \
        --env tts --out /opt/voxkitchen/schemas_tts.json

# ============================================================
# fish-speech-env: isolated torch 2.8 + numpy 2.1 stack for fish-speech.
# Minimal — only installs [tts-fish-speech]. Other operators (resample,
# vad, pack, ...) are NOT available here; the runner dispatches those to
# core/asr/tts via the normal cross-env subprocess mechanism.
# ============================================================
FROM core-env AS fish-speech-env

RUN uv venv /opt/voxkitchen/envs/fish-speech --python 3.11

# Install torch 2.8 from cu124 first so the constraint is satisfied
# before fish-speech's own deps try to pull it.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python \
        -c /app/docker/constraints/fish-speech.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.8.0 torchaudio==2.8.0

# fish-speech proper. Installed from git directly here rather than via a
# `[tts-fish-speech]` extra in pyproject.toml — PyPI refuses to accept any
# uploaded distribution whose metadata declares a PEP 440 direct reference
# (`pkg @ git+...`), so the published wheel cannot carry this URL.
# Pinned to the Fish-Speech 2.0 commit validated by the Docker smoke test.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python \
        -c /app/docker/constraints/fish-speech.txt \
        "fish-speech @ git+https://github.com/fishaudio/fish-speech.git@3dd1f85c402ee6f0a17c2971d3b0dd8d881ca139"

# Pin voxkitchen's editable link to /app source.
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python -e . --no-deps

RUN /opt/voxkitchen/envs/fish-speech/bin/python scripts/warmup_models.py --group fish-speech || true
RUN /opt/voxkitchen/envs/fish-speech/bin/python -m voxkitchen.runtime.dump_schemas \
        --env fish-speech --out /opt/voxkitchen/schemas_fish-speech.json

# ============================================================
# Single-env targets. Each one:
#   * merges core + <env> schemas into op_schemas.json
#   * runs `vkit doctor --expect <env>` as a build-time smoke test
#   * flips permissions so non-root users can read site-packages + write cache
#
# Users who only need one capability (e.g. "just diarization") pull the
# matching tag instead of :latest. Cross-env pipelines still work — if a
# stage's operator maps to an env not in the image, you get a clear
# "env not present" error instead of silent fallback.
# ============================================================

FROM asr-env AS asr
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        /opt/voxkitchen/schemas_asr.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core
RUN /opt/voxkitchen/envs/asr/bin/vkit  doctor --expect asr
RUN chmod -R a+rX /opt/voxkitchen && chmod -R a+rwX /opt/voxkitchen/model_cache
LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen asr" \
      org.opencontainers.image.description="core + ASR (faster-whisper, funasr, qwen3, forced alignment)"
ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]


FROM diarize-env AS diarize
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        /opt/voxkitchen/schemas_diarize.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core
RUN /opt/voxkitchen/envs/diarize/bin/vkit doctor --expect diarize
RUN chmod -R a+rX /opt/voxkitchen && chmod -R a+rwX /opt/voxkitchen/model_cache
LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen diarize" \
      org.opencontainers.image.description="core + pyannote speaker diarization"
ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]


FROM tts-env AS tts
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        /opt/voxkitchen/schemas_tts.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core
RUN /opt/voxkitchen/envs/tts/bin/vkit  doctor --expect tts
RUN chmod -R a+rX /opt/voxkitchen && chmod -R a+rwX /opt/voxkitchen/model_cache
LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen tts" \
      org.opencontainers.image.description="core + TTS engines (kokoro, ChatTTS, CosyVoice)"
ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]


FROM fish-speech-env AS fish-speech
RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        /opt/voxkitchen/schemas_fish-speech.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json
RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core
RUN /opt/voxkitchen/envs/fish-speech/bin/vkit doctor --expect fish-speech
RUN chmod -R a+rX /opt/voxkitchen && chmod -R a+rwX /opt/voxkitchen/model_cache
LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen fish-speech" \
      org.opencontainers.image.description="core + fish-speech (isolated torch 2.8 stack)"
ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]


# ============================================================
# latest (target): all envs in one image via COPY --from merging.
# Independent env branches are assembled here so BuildKit can parallelize
# their construction.
# ============================================================
FROM core-env AS latest

COPY --from=asr-env          /opt/voxkitchen/envs/asr          /opt/voxkitchen/envs/asr
COPY --from=asr-env          /opt/voxkitchen/schemas_asr.json  /opt/voxkitchen/
COPY --from=asr-env          /opt/voxkitchen/warmup_asr.json   /opt/voxkitchen/
COPY --from=asr-env          /opt/voxkitchen/model_cache       /opt/voxkitchen/model_cache

COPY --from=diarize-env      /opt/voxkitchen/envs/diarize          /opt/voxkitchen/envs/diarize
COPY --from=diarize-env      /opt/voxkitchen/schemas_diarize.json  /opt/voxkitchen/
COPY --from=diarize-env      /opt/voxkitchen/warmup_diarize.json   /opt/voxkitchen/
COPY --from=diarize-env      /opt/voxkitchen/model_cache           /opt/voxkitchen/model_cache

COPY --from=tts-env          /opt/voxkitchen/envs/tts          /opt/voxkitchen/envs/tts
COPY --from=tts-env          /opt/voxkitchen/schemas_tts.json  /opt/voxkitchen/
COPY --from=tts-env          /opt/voxkitchen/warmup_tts.json   /opt/voxkitchen/
COPY --from=tts-env          /opt/voxkitchen/model_cache       /opt/voxkitchen/model_cache

COPY --from=fish-speech-env  /opt/voxkitchen/envs/fish-speech          /opt/voxkitchen/envs/fish-speech
COPY --from=fish-speech-env  /opt/voxkitchen/schemas_fish-speech.json  /opt/voxkitchen/
COPY --from=fish-speech-env  /opt/voxkitchen/warmup_fish-speech.json   /opt/voxkitchen/
COPY --from=fish-speech-env  /opt/voxkitchen/model_cache               /opt/voxkitchen/model_cache

RUN /opt/voxkitchen/envs/core/bin/python -m voxkitchen.runtime.merge_schemas \
        /opt/voxkitchen/schemas_core.json \
        /opt/voxkitchen/schemas_asr.json \
        /opt/voxkitchen/schemas_diarize.json \
        /opt/voxkitchen/schemas_tts.json \
        /opt/voxkitchen/schemas_fish-speech.json \
        --schemas-out /opt/voxkitchen/op_schemas.json \
        --env-map-out /opt/voxkitchen/op_env_map.json

RUN /opt/voxkitchen/envs/core/bin/vkit doctor --expect core
RUN /opt/voxkitchen/envs/asr/bin/vkit  doctor --expect asr
RUN /opt/voxkitchen/envs/diarize/bin/vkit doctor --expect diarize
RUN /opt/voxkitchen/envs/tts/bin/vkit  doctor --expect tts
RUN /opt/voxkitchen/envs/fish-speech/bin/vkit doctor --expect fish-speech

RUN chmod -R a+rX /opt/voxkitchen && chmod -R a+rwX /opt/voxkitchen/model_cache

LABEL org.opencontainers.image.source="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.url="https://github.com/XqFeng-Josie/VoxKitchen" \
      org.opencontainers.image.documentation="https://github.com/XqFeng-Josie/VoxKitchen/blob/main/docs/docker-build.md" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.title="VoxKitchen latest" \
      org.opencontainers.image.description="all five envs merged (core / asr / diarize / tts / fish-speech) — for cross-cluster pipelines"

ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]
