#!/usr/bin/env bash
# Thin wrapper around `docker build` that:
#   * picks HF_TOKEN out of ./.env (so pyannote / other gated models get
#     baked into the image instead of downloading at runtime), and
#   * defaults to the `latest` target with `-t voxkitchen:latest`.
#
# Usage:
#   scripts/vkit-build.sh                # build latest (all envs)
#   scripts/vkit-build.sh slim           # CPU core only
#   scripts/vkit-build.sh asr            # core + ASR (no diarize)
#   scripts/vkit-build.sh diarize        # core + pyannote only
#   scripts/vkit-build.sh tts            # core + TTS (kokoro/chattts/cosyvoice)
#   scripts/vkit-build.sh fish-speech    # core + fish-speech (isolated torch 2.8)
#   VKIT_TARGET=slim scripts/vkit-build.sh
#   VKIT_TAG=voxkitchen:dev scripts/vkit-build.sh
#
# No args fall through to `docker build` — pass extra docker flags after
# the target name, e.g.
#   scripts/vkit-build.sh latest --progress=plain --no-cache

set -euo pipefail

TARGET="${VKIT_TARGET:-${1:-latest}}"
if [[ "${1:-}" == "$TARGET" ]]; then
    shift
fi
TAG="${VKIT_TAG:-voxkitchen:${TARGET}}"

build_args=()

# Extract HF_TOKEN from .env if present and non-empty. Do not export it
# into our own environment; pass it directly as --build-arg.
if [[ -f .env ]]; then
    hf_token=$(grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'") || true
    if [[ -n "${hf_token:-}" ]]; then
        build_args+=(--build-arg "HF_TOKEN=${hf_token}")
        echo "[vkit-build] using HF_TOKEN from .env (pyannote model will be baked in)"
    else
        echo "[vkit-build] .env present but no HF_TOKEN — pyannote will download at runtime"
    fi
else
    echo "[vkit-build] no .env found — pyannote will download at runtime"
fi

exec docker build \
    --target "${TARGET}" \
    -f docker/Dockerfile \
    -t "${TAG}" \
    "${build_args[@]}" \
    "$@" \
    .
