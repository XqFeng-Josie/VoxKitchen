# ============================================================
# VoxKitchen — batteries-included Docker image
#
# Includes all 51 operators, system deps (FFmpeg, espeak-ng),
# and GPU support via CUDA 12.4.
#
# Build:
#   docker build -t voxkitchen .
#
# Run a pipeline:
#   docker run --rm -v /data/raw_audio:/data voxkitchen vkit run pipeline.yaml
#
# Run with GPU:
#   docker run --rm --gpus all -v /data/raw_audio:/data voxkitchen vkit run pipeline.yaml
#
# Interactive:
#   docker run --rm -it voxkitchen bash
#
# Run tests:
#   docker run --rm voxkitchen pytest tests/unit/operators/ -v -m "not gpu"
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

# PyPI-based extras
RUN pip install -e ".[audio,segment,quality,pack,asr,whisper,pitch,dnsmos,funasr,classify,gender,codec,align,tts-kokoro,tts-chattts,tts-cosyvoice,viz,viz-panel]" \
        pytest pytest-cov scipy 2>&1 \
    || echo "WARN: some extras failed, continuing..."

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

# ---- default: start interactive CLI ----
ENTRYPOINT ["vkit"]
CMD ["--help"]
