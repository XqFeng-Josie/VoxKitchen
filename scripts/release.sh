#!/usr/bin/env bash
# One-command release wrapper — see RELEASING.md for what each step does.
#
# Usage:
#   scripts/release.sh <version>
#
# Example:
#   scripts/release.sh 0.2.0
#
# The script is interactive: it prompts before each destructive step
# (tag, git push, docker push). Ctrl-C out at any prompt to abort.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    echo "usage: $0 <version>   (e.g. 0.2.0)" >&2
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

if git rev-parse "$TAG" >/dev/null 2>&1; then
    die "tag $TAG already exists locally."
fi

if ! grep -q "^## \[${VERSION}\]" CHANGELOG.md; then
    die "CHANGELOG.md has no '## [${VERSION}]' section — prepare it first (see RELEASING.md §2)."
fi

log "Running tests (not slow / not gpu)"
pytest -q -m "not slow and not gpu" >/dev/null || die "tests failed — fix before releasing."

log "Pre-flight ok."

# ---------------------------------------------------------------------------
# 2. Tag
# ---------------------------------------------------------------------------

log "Creating annotated tag $TAG on HEAD ($(git rev-parse --short HEAD))"
confirm "create tag $TAG?"
git tag -a "$TAG" -m "Release $TAG"

# ---------------------------------------------------------------------------
# 3. Push
# ---------------------------------------------------------------------------

log "Pushing main and $TAG to origin"
confirm "push to origin?"
git push origin main
git push origin "$TAG"

# ---------------------------------------------------------------------------
# 4. Docker build + push
# ---------------------------------------------------------------------------

if ! command -v docker >/dev/null 2>&1; then
    warn "docker CLI not found — skipping image publish."
    warn "publish manually later: see RELEASING.md §5."
    exit 0
fi

log "Docker build + push ${#TARGETS[@]} targets (smallest → largest)"
echo "   targets: ${TARGETS[*]}"
echo "   rolling tag: ${GHCR_IMAGE}:<target>"
echo "   pinned tag:  ${GHCR_IMAGE}:<target>-${VERSION}"
confirm "build and push now? (this takes 1-2 hours for all six)"

for target in "${TARGETS[@]}"; do
    log "[$target] building"
    docker build --target "$target" -f docker/Dockerfile -t "voxkitchen:$target" .

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

if command -v gh >/dev/null 2>&1; then
    echo "   to create the release page now:"
    echo "     gh release create $TAG --title '$TAG' --notes-from-tag"
    echo "   or paste the CHANGELOG [${VERSION}] section into:"
else
    echo "   gh CLI not found. Create the release manually at:"
fi
echo "     https://github.com/XqFeng-Josie/VoxKitchen/releases/new?tag=${TAG}"

log "Release $TAG published."
echo "Don't forget: update CHANGELOG's [Unreleased] section for the next cycle."
