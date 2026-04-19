# syntax=docker/dockerfile:1.6
# ============================================================
# VoxKitchen: one image, multiple Python envs.
#
# Two build targets:
#   --target slim    core env only   (~3 GB, CPU, no ASR/TTS)
#   --target latest  core + asr + tts (~18 GB, GPU, everything)
#
# Usage:
#   docker build --target slim   -t voxkitchen:slim   -f docker/Dockerfile .
#   docker build --target latest -t voxkitchen:latest -f docker/Dockerfile .
#
# With pyannote diarization baked in (needs HuggingFace token):
#   docker build --target latest --build-arg HF_TOKEN=hf_xxx -t voxkitchen:latest -f docker/Dockerfile .
#
# See docs/architecture/multi-env.md for the design.
# ============================================================

# Global build args (available in every stage that re-declares them).
ARG HF_TOKEN=""

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS base

# uv is ~10x faster than pip and handles per-venv deps cleanly.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    # Scope the pretend version to voxkitchen only. A bare
    # SETUPTOOLS_SCM_PRETEND_VERSION leaks into every setuptools-scm
    # build in the venv — observed breaking pyopenjtalk (misaki[ja] dep)
    # because its wheel then declared 0.0.0.dev0 != the requested 0.4.1.
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXKITCHEN=0.0.0.dev0 \
    UV_NO_CACHE=1 \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject + a stub package first so pip install is cacheable even
# when source files change. Real source is copied at the end of each env
# stage and reinstalled with --no-deps.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p voxkitchen && \
    echo '__version__ = "0.0.0.dev0"' > voxkitchen/_version.py && \
    echo '' > voxkitchen/__init__.py

COPY docker/constraints/ /app/docker/constraints/

# ============================================================
# core-env: CPU torch, lightweight deps. Parent env of any build.
# ============================================================
FROM base AS core-env

RUN uv venv /opt/voxkitchen/envs/core --python 3.11

# CPU PyTorch from the dedicated index (uv respects --index-url per-package).
RUN uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

# Core package (no extras yet). MUST succeed.
RUN uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        -e .

# Core extras. [gender] is intentionally excluded everywhere — inaSpeechSegmenter
# pulls tensorflow[and-cuda] which fights every other dep cluster. The
# gender_classify operator still registers and works via method=f0 / method=speechbrain.
RUN uv pip install --python /opt/voxkitchen/envs/core/bin/python \
        -c /app/docker/constraints/core.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,viz-panel]"

# (wespeaker is pulled transitively by the [speaker] extras above — no
# separate install needed.)

# Full source (overwrites the stub). Reinstall with --no-deps so the resolver
# doesn't touch already-installed extras.
COPY . .
RUN uv pip install --python /opt/voxkitchen/envs/core/bin/python -e . --no-deps

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
RUN uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

# ASR env inherits everything core has + ASR-specific extras. diarize
# is deliberately NOT here — it lives in its own env so a ":diarize"
# image can ship without the full ASR stack.
RUN uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,viz-panel,asr,whisper,funasr,align]"

# wenet_asr is intentionally NOT in EXPECTED_OPERATORS.asr — its upstream
# tooling breaks often and we don't want one flaky dep to block the image.
# If the install fails, the operator just doesn't register; users who
# need it can install wenet themselves.
RUN uv pip install --python /opt/voxkitchen/envs/asr/bin/python \
        -c /app/docker/constraints/asr.txt \
        "wenet @ git+https://github.com/wenet-e2e/wenet.git" \
        || echo "WARN: wenet install failed — wenet_asr operator will be unavailable"

RUN uv pip install --python /opt/voxkitchen/envs/asr/bin/python -e . --no-deps

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

RUN uv pip install --python /opt/voxkitchen/envs/diarize/bin/python \
        -c /app/docker/constraints/diarize.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

RUN uv pip install --python /opt/voxkitchen/envs/diarize/bin/python \
        -c /app/docker/constraints/diarize.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,viz-panel,diarize]"

RUN uv pip install --python /opt/voxkitchen/envs/diarize/bin/python -e . --no-deps

RUN HF_TOKEN=${HF_TOKEN} /opt/voxkitchen/envs/diarize/bin/python scripts/warmup_models.py --group diarize || true
RUN /opt/voxkitchen/envs/diarize/bin/python -m voxkitchen.runtime.dump_schemas \
        --env diarize --out /opt/voxkitchen/schemas_diarize.json

# ============================================================
# tts-env: layered on core-env (independent of asr)
# ============================================================
FROM core-env AS tts-env

RUN uv venv /opt/voxkitchen/envs/tts --python 3.11

RUN uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.4.1 torchaudio==2.4.1

RUN uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        -c /app/docker/constraints/tts.txt \
        -e ".[audio,segment,quality,pack,pitch,dnsmos,classify,enhance,codec,speaker,viz,viz-panel,tts-kokoro,tts-chattts,tts-cosyvoice]"

# misaki (pulled by kokoro) lazy-downloads the spacy English model at first
# TTS call via `uv pip install`. In a non-root runtime that lazy install
# fails (can't write to /opt/voxkitchen/envs/*). Install it now so every
# runtime call is a cache hit.
RUN uv pip install --python /opt/voxkitchen/envs/tts/bin/python \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# [tts-fish-speech] is deliberately omitted: fish-speech HEAD pins
# torch==2.8.0, incompatible with this env's torch 2.4.1 stack (ChatTTS /
# CosyVoice / kokoro all validated against 2.4). Promoting fish-speech
# back into the image requires either (a) a new env with torch 2.8, or
# (b) upstream accepting our torch 2.4 range. The tts_fish_speech
# operator source stays in the repo; users can `pip install fish-speech`
# at runtime in a throwaway container if they need it.

RUN uv pip install --python /opt/voxkitchen/envs/tts/bin/python -e . --no-deps

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
RUN uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python \
        -c /app/docker/constraints/fish-speech.txt \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple \
        torch==2.8.0 torchaudio==2.8.0

# fish-speech proper + voxkitchen base deps.
RUN uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python \
        -c /app/docker/constraints/fish-speech.txt \
        -e ".[tts-fish-speech]"

# Pin voxkitchen's editable link to /app source.
RUN uv pip install --python /opt/voxkitchen/envs/fish-speech/bin/python -e . --no-deps

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

ENV PATH=/opt/voxkitchen/envs/core/bin:$PATH \
    VKIT_ENV=core
ENTRYPOINT ["vkit"]
CMD ["--help"]
