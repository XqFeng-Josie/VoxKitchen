# ============================================================
# VoxKitchen — batteries-included Docker image
#
# Includes all 51 operators, system deps (FFmpeg, espeak-ng),
# and GPU support via CUDA 12.4.
#
# Build:
#   docker build -t voxkitchen .
#
# Quick demo:
#   docker run --rm voxkitchen run examples/pipelines/demo-no-asr.yaml
#
# Run your own pipeline:
#   docker run --rm -v /data/raw_audio:/data voxkitchen run pipeline.yaml
#
# Run with GPU:
#   docker run --rm --gpus all -v /data/raw_audio:/data voxkitchen run pipeline.yaml
#
# Interactive shell:
#   docker run --rm -it --entrypoint bash voxkitchen
# ============================================================

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0.dev0

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        pkg-config \
        ffmpeg \
        libavformat-dev \
        libavcodec-dev \
        libavutil-dev \
        libavfilter-dev \
        libswscale-dev \
        libswresample-dev \
        espeak-ng \
        sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- pip deps (layer-cached) ----
COPY pyproject.toml README.md LICENSE ./
# Minimal package skeleton so pip install -e works
RUN mkdir -p voxkitchen && \
    echo '__version__ = "0.0.0.dev0"' > voxkitchen/_version.py && \
    echo '' > voxkitchen/__init__.py

# Core deps MUST succeed — no fallback
RUN pip install -e .

# Optional PyPI-based extras — install individually so one failure
# does not block the rest
RUN for extra in audio segment quality pack asr whisper pitch dnsmos \
        funasr classify gender codec align \
        tts-kokoro tts-chattts tts-cosyvoice viz viz-panel; do \
        pip install -e ".[$extra]" 2>&1 \
        || echo "WARN: [$extra] install failed, skipping"; \
    done

# Test deps
RUN pip install pytest pytest-cov scipy

# Git-based deps — separate layers for caching
RUN pip install "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git" 2>&1 \
    || echo "WARN: wespeaker install failed, skipping"
RUN pip install "wenet @ git+https://github.com/wenet-e2e/wenet.git" 2>&1 \
    || echo "WARN: wenet install failed, skipping"
RUN pip install "fish-speech @ git+https://github.com/fishaudio/fish-speech.git" 2>&1 \
    || echo "WARN: fish-speech install failed, skipping"

# Enhance (needs system FFmpeg libs installed above)
RUN pip install "deepfilternet>=0.5" 2>&1 \
    || echo "WARN: deepfilternet install failed, skipping"

# ---- copy full source & install ----
COPY . .
RUN pip install -e . --no-deps

# ---- default: vkit CLI ----
ENTRYPOINT ["vkit"]
CMD ["--help"]
