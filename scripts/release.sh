#!/usr/bin/env bash
# One-command release wrapper — see RELEASING.md for what each step does.
#
# Usage:
#   scripts/release.sh <version> [--replace-unpublished]
#
# Example:
#   scripts/release.sh 0.2.0
#
# The script is interactive: it prompts before each destructive step
# (tag, git push, docker push). Ctrl-C out at any prompt to abort.
# Use --replace-unpublished only when replacing a tag/image set that has not
# been consumed by users.

set -euo pipefail

VERSION="${1:-}"
REPLACE_UNPUBLISHED=false
if [[ -z "$VERSION" ]]; then
    echo "usage: $0 <version> [--replace-unpublished]   (e.g. 0.2.0)" >&2
    exit 2
fi
if [[ "${2:-}" == "--replace-unpublished" ]]; then
    REPLACE_UNPUBLISHED=true
elif [[ -n "${2:-}" ]]; then
    echo "error: unknown option '${2:-}'" >&2
    exit 2
fi
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "error: version must be MAJOR.MINOR.PATCH (e.g. 0.2.0), got '$VERSION'" >&2
    exit 2
fi
TAG="v${VERSION}"

GHCR_IMAGE="ghcr.io/xqfeng-josie/voxkitchen"
TARGETS=(slim diarize fish-speech asr tts latest)

log()  { printf "\n\033[1;34m==>\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!!\033[0m %s\n" "$*" >&2; }
die()  { printf "\033[1;31merror:\033[0m %s\n" "$*" >&2; exit 1; }

confirm() {
    local prompt="$1"
    read -r -p "$prompt [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]] || die "aborted."
}

write_release_notes() {
    local notes_file="/tmp/voxkitchen-${TAG}-notes.md"
    awk -v version="$VERSION" '
        $0 ~ "^## \\[" version "\\]" { in_section=1; next }
        /^## \[/ { if (in_section) exit }
        in_section { print }
    ' CHANGELOG.md > "$notes_file"

    if [[ ! -s "$notes_file" ]]; then
        warn "could not extract CHANGELOG [${VERSION}] notes; fill the GitHub Release description manually."
    fi

    printf "%s" "$notes_file"
}

configure_docker_workspace() {
    DOCKER_WORK_DIR="${VKIT_DOCKER_WORK_DIR:-$PWD/.docker}"
    DOCKER_CONFIG="${DOCKER_CONFIG:-$DOCKER_WORK_DIR/config}"
    TMPDIR="${TMPDIR:-$DOCKER_WORK_DIR/tmp}"
    BUILDX_CONFIG="${BUILDX_CONFIG:-$DOCKER_WORK_DIR/buildx}"
    XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DOCKER_WORK_DIR/cache}"
    export DOCKER_WORK_DIR DOCKER_CONFIG TMPDIR BUILDX_CONFIG XDG_CACHE_HOME

    mkdir -p "$DOCKER_CONFIG" "$TMPDIR" "$BUILDX_CONFIG" "$XDG_CACHE_HOME"
}

# ---------------------------------------------------------------------------
# 1. Pre-flight
# ---------------------------------------------------------------------------

log "Pre-flight checks"

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "working tree is not clean — commit or stash first."
fi

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != "main" ]]; then
    warn "you are on branch '$branch', not 'main'."
    confirm "continue anyway?"
fi

git fetch origin main
if ! git diff --quiet "origin/main" -- .; then
    warn "local main differs from origin/main."
    confirm "continue anyway?"
fi

LOCAL_TAG_EXISTS=false
REMOTE_TAG_EXISTS=false
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    LOCAL_TAG_EXISTS=true
fi
if git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1; then
    REMOTE_TAG_EXISTS=true
fi
if [[ "$LOCAL_TAG_EXISTS" == true || "$REMOTE_TAG_EXISTS" == true ]]; then
    if [[ "$REPLACE_UNPUBLISHED" != true ]]; then
        die "tag $TAG already exists. Use --replace-unpublished only if nobody has consumed it."
    fi
    warn "tag $TAG already exists; this will replace an unpublished release tag."
    confirm "replace unpublished $TAG?"
fi

if ! grep -q "^## \[${VERSION}\]" CHANGELOG.md; then
    die "CHANGELOG.md has no '## [${VERSION}]' section — prepare it first (see RELEASING.md §2)."
fi

log "Running local CI checks"
scripts/check-ci.sh || die "local CI checks failed — fix before releasing."

log "Pre-flight ok."

# ---------------------------------------------------------------------------
# 2. Tag
# ---------------------------------------------------------------------------

log "Creating annotated tag $TAG on HEAD ($(git rev-parse --short HEAD))"
confirm "create tag $TAG?"
if [[ "$LOCAL_TAG_EXISTS" == true ]]; then
    git tag -d "$TAG"
fi
git tag -a "$TAG" -m "Release $TAG"

# ---------------------------------------------------------------------------
# 3. Push
# ---------------------------------------------------------------------------

log "Pushing main and $TAG to origin"
echo "   Once the tag lands on origin, .github/workflows/publish.yml will"
echo "   build the wheel + sdist and upload to PyPI via Trusted Publishing."
echo "   Make sure the pypi.org pending publisher is configured first"
echo "   (see RELEASING.md §5)."
confirm "push to origin?"
git push origin main
if [[ "$REMOTE_TAG_EXISTS" == true ]]; then
    git push --force origin "$TAG"
else
    git push origin "$TAG"
fi
echo "   PyPI workflow run:"
echo "     https://github.com/XqFeng-Josie/VoxKitchen/actions/workflows/publish.yml"

# ---------------------------------------------------------------------------
# 4. Docker build + push
# ---------------------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    warn "docker CLI not found — skipping image publish."
    warn "publish Docker images manually later: see RELEASING.md §6."
    exit 0
fi

configure_docker_workspace

log "Docker build + push ${#TARGETS[@]} targets"
echo "   targets: ${TARGETS[*]}"
echo "   rolling tag: ${GHCR_IMAGE}:<target>"
echo "   pinned tag:  ${GHCR_IMAGE}:<target>-${VERSION}"
echo "   Docker work dir: $DOCKER_WORK_DIR"
echo "   DOCKER_CONFIG:   $DOCKER_CONFIG"
echo "   TMPDIR:          $TMPDIR"
echo "   BUILDX_CONFIG:   $BUILDX_CONFIG"
echo "   XDG_CACHE_HOME:  $XDG_CACHE_HOME"
echo "   note: Docker image layers still use the daemon data-root (often /var/lib/docker)."
confirm "build and push now? (this takes 1-2 hours for all six)"

for target in "${TARGETS[@]}"; do
    log "[$target] building"
    docker build \
        --build-arg "VOXKITCHEN_VERSION=${VERSION}" \
        --target "$target" \
        -f docker/Dockerfile \
        -t "voxkitchen:$target" \
        .

    log "[$target] tagging"
    docker tag "voxkitchen:$target" "${GHCR_IMAGE}:${target}"
    docker tag "voxkitchen:$target" "${GHCR_IMAGE}:${target}-${VERSION}"

    log "[$target] pushing"
    docker push "${GHCR_IMAGE}:${target}"
    docker push "${GHCR_IMAGE}:${target}-${VERSION}"
done

# ---------------------------------------------------------------------------
# 5. GitHub Release
# ---------------------------------------------------------------------------

log "GitHub Release"

NOTES_FILE="$(write_release_notes)"
if command -v gh >/dev/null 2>&1; then
    echo "   to create the release page now:"
    echo "     gh release create $TAG -t '$TAG' -F '$NOTES_FILE'"
    echo "   if the release page already exists, edit its description in the web UI:"
    echo "     https://github.com/XqFeng-Josie/VoxKitchen/releases/tag/${TAG}"
    echo "   or paste the CHANGELOG [${VERSION}] section into:"
else
    echo "   gh CLI not found. Create the release manually at:"
fi
echo "     https://github.com/XqFeng-Josie/VoxKitchen/releases/new?tag=${TAG}"

log "Release $TAG published."
echo "Don't forget: update CHANGELOG's [Unreleased] section for the next cycle."
