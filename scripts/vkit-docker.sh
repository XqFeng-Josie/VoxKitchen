#!/usr/bin/env bash
# Thin wrapper around `docker run voxkitchen:<tag>` that preserves file
# ownership on bind mounts and wires up the canonical volumes.
#
# Usage:
#   scripts/vkit-docker.sh run examples/pipelines/demo-no-asr.yaml
#   scripts/vkit-docker.sh doctor
#   VKIT_TAG=slim scripts/vkit-docker.sh run ...
#   VKIT_GPU=0 scripts/vkit-docker.sh run ...             # force CPU-only
#   VKIT_ENV_FILE=.env scripts/vkit-docker.sh ...         # pass env file
#   VKIT_MOUNT="examples/pipelines/smoke-diarize.yaml" \  # extra bind mount
#       scripts/vkit-docker.sh run examples/pipelines/smoke-diarize.yaml
#
# What this buys you over a raw `docker run`:
#   * --user maps container UID to your host UID, so files in work/ are
#     yours to delete without sudo.
#   * HOME=/tmp keeps non-root python tooling happy (huggingface, torch).
#   * ./work and ./data are bind-mounted by convention.
#   * --gpus all is added when nvidia-smi is available unless VKIT_GPU=0.
#   * ./.env is auto-included when it exists (HF_TOKEN for gated models).
#   * VKIT_MOUNT can be a space-separated list of host paths to bind
#     read-only at the same path inside the container — handy for
#     pipeline YAMLs that were added after the image was built.

set -euo pipefail

TAG="${VKIT_TAG:-latest}"
IMAGE="${VKIT_IMAGE:-voxkitchen:${TAG}}"

gpu_args=()
if [[ "${VKIT_GPU:-auto}" != "0" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    gpu_args=(--gpus all)
fi

env_args=()
env_file="${VKIT_ENV_FILE:-.env}"
if [[ -f "$env_file" ]]; then
    env_args=(--env-file "$env_file")
fi

# Extra bind mounts (whitespace-separated). Each path is mounted
# read-only at the same absolute path inside the container — so YAML
# files referencing host-relative paths keep working.
mount_args=()
if [[ -n "${VKIT_MOUNT:-}" ]]; then
    for p in ${VKIT_MOUNT}; do
        abs=$(realpath "$p")
        mount_args+=(-v "${abs}:${abs}:ro")
    done
fi

# Ensure host work/ exists BEFORE docker bind-mounts it. If it doesn't,
# docker daemon creates it as root, and a non-root --user then can't
# write into it.
mkdir -p ./work

exec docker run --rm \
    "${gpu_args[@]}" \
    "${env_args[@]}" \
    "${mount_args[@]}" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$(pwd)/work":/app/work \
    -v "$(pwd)/data":/data \
    "${IMAGE}" \
    "$@"
