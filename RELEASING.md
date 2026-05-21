# Releasing VoxKitchen

This document describes the process for cutting a new release. The same
procedure is automated by [`scripts/release.sh`](scripts/release.sh) —
read this doc to understand what the script does; run the script to
actually do it.

## Version number

We follow [semver](https://semver.org/) within pre-alpha constraints
(0.x.y until the API settles):

- **MAJOR (0.x → 1.0)**: API is considered stable. Not planned yet.
- **MINOR (0.x.0)**: new features, new operators, new Docker envs, or
  notable behavioral changes.
- **PATCH (0.x.y)**: bug fixes and docs that don't change behavior.

The version string is derived from the latest git tag via
[hatch-vcs](https://github.com/ofek/hatch-vcs) (see `[tool.hatch.version]`
in `pyproject.toml`). We do **not** hand-maintain a version constant.

## One-command release

```bash
scripts/release.sh 0.2.0
```

The script runs through the checklist below. It will prompt you before
each destructive step (tag, push, Docker push).

## Checklist (what the script does, in order)

### 1. Pre-flight

- Working tree must be clean (no uncommitted changes).
- Current branch must be `main` and up-to-date with `origin/main`.
- Local CI checks must pass: `scripts/check-ci.sh`.
- `CHANGELOG.md` must have a section matching the new version (see below).

### 2. CHANGELOG

Move entries from `[Unreleased]` into a new dated section. Example:

```markdown
## [Unreleased]

## [0.2.0] — 2026-05-18

### Added
- new feature X

### Changed
- behavior Y

### Fixed
- bug Z

### Known limitations
- (copy forward any still-relevant items)
```

Commit: `chore(release): prepare v0.2.0`

### 3. Tag

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
```

The tag message should summarize the release in a paragraph or two.
Users see this via `git show v0.2.0` and GitHub Release pages often
populate from it.

### 4. Push

```bash
git push origin main
git push origin v0.2.0
```

Main commit first, then tag. This order is important — the tag points
to a commit that must already be on the remote.

### 5. PyPI publish (automated)

Pushing the `v*` tag triggers
[`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds
the wheel + sdist and uploads to <https://pypi.org/project/voxkitchen/> via
PyPI Trusted Publishing (OIDC, no stored API tokens). No local action is
needed — watch the workflow under the Actions tab on GitHub.

To verify before a release, use the workflow's manual dispatch:

```
GitHub → Actions → publish → Run workflow → target: testpypi
```

This uploads to <https://test.pypi.org/project/voxkitchen/>.

#### First-time PyPI setup (do once before the first release)

PyPI Trusted Publishing needs a "pending publisher" registered on PyPI before
the project exists. Do this once:

1. Sign in to <https://pypi.org> and go to **Your projects → Publishing**.
2. Click **Add a new pending publisher** and fill in:
   - PyPI Project Name: `voxkitchen`
   - Owner: `XqFeng-Josie`
   - Repository name: `VoxKitchen`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. Repeat on <https://test.pypi.org> with environment name `testpypi`.

After the first successful publish, the pending publisher is converted to a
regular trusted publisher and no further PyPI-side setup is needed.

### 6. Docker images

`scripts/release.sh` builds and pushes each of the six targets in a fixed
order and tags each with both the rolling
`:<target>` and the pinned `:<target>-<version>`. It also passes
`VOXKITCHEN_VERSION=<version>` into the Docker build so package metadata
inside the image matches the release tag. See
[`scripts/release.sh`](scripts/release.sh) for the exact loop.

Before Docker build/push, the script defaults Docker client scratch paths to
`./.docker`:

- `DOCKER_CONFIG=./.docker/config`
- `TMPDIR=./.docker/tmp`
- `BUILDX_CONFIG=./.docker/buildx`
- `XDG_CACHE_HOME=./.docker/cache`

Set `VKIT_DOCKER_WORK_DIR=/data2/.../.docker` to use a different base
directory. This does not move Docker image layers; if `/var/lib/docker` is
filling `/`, move Docker daemon `data-root` as described in
[`docs/docker-build.md`](docs/docker-build.md).

**Prerequisites**:

- Logged in to GHCR with the same Docker config the release script will use:
  `mkdir -p .docker/config && printf '%s' "$GHCR_TOKEN" | DOCKER_CONFIG="$PWD/.docker/config" docker login ghcr.io -u xqfeng-josie --password-stdin`.
  The PAT needs `write:packages` scope (classic PAT from
  https://github.com/settings/tokens/new).
- HF_TOKEN in `./.env` if you want to bake pyannote into the `asr` and
  `diarize` images (pipeline works without it, but first run will
  download ~80 MB of pyannote weights).

Build+push of all six targets on a well-provisioned machine takes
~1-2 hours (latest is the bottleneck at ~123 GB).

If the Docker images were already rebuilt and pushed manually for this exact
commit, do **not** run `scripts/release.sh` again; it will rebuild and push
all six images. Instead, run the pre-flight checks, commit, tag, and push
manually using steps 3, 4, and 6.

### 7. GitHub Release

The git tag is not the same as a public Release on GitHub. After the
tag is pushed, create the Release page to attach release notes:

```bash
# CLI (requires `gh` installed and authenticated)
VERSION=0.2.0
TAG="v${VERSION}"
NOTES="/tmp/voxkitchen-${TAG}-notes.md"

awk -v version="$VERSION" '
  $0 ~ "^## \\[" version "\\]" { in_section=1; next }
  /^## \[/ { if (in_section) exit }
  in_section { print }
' CHANGELOG.md > "$NOTES"

gh release create "$TAG" -t "$TAG" -F "$NOTES"
```

Or via the web UI:
https://github.com/XqFeng-Josie/VoxKitchen/releases/new?tag=v0.2.0

Paste the relevant CHANGELOG section into the "Description" field and publish.
If the Release page already exists, edit the description from the web UI; older
`gh` builds may not include `gh release edit`.

### 8. Verify

One last sanity check against the published artifact:

```bash
# Verify PyPI publish picked up the tag
pipx install --force voxkitchen==X.Y.Z
vkit --help

# Pull a published image and run doctor against it
vkit docker pull --tag slim
vkit docker doctor --tag slim --expect core

# Expected: 36 operators available in core env, 0 missing.
```

## Why no CI for Docker publish

GitHub Actions default runners have ~14 GB of free disk. Our `:asr`,
`:tts`, `:fish-speech`, and `:latest` images are 40-123 GB each and will not fit. Full
Docker publish CI requires either self-hosted runners or GitHub's
larger-runner add-ons.

Non-Docker CI (lint, unit tests, typecheck) runs on every push via
`.github/workflows/ci.yml` — that's the frequent regression gate.

## Replacing an unpublished release

Only do this before anyone has consumed the tag or Docker images. For an
unpublished first release, it is acceptable to replace the tag and pinned image
tags so the public `v0.2.0` points at the corrected commit:

```bash
scripts/release.sh 0.2.0 --replace-unpublished
```

This force-pushes only the release tag when it already exists. It still pushes
`main` normally and rebuilds all Docker targets. Do not use this path after the
release has external users; publish a patch release instead.

## Yanking a release

If a pushed release has a critical bug:

1. **Publish a patch release** with the fix (`vX.Y.Z+1`) — this is the
   normal path for users pinned to a release tag.
2. **Mark the broken release as such** in `CHANGELOG.md` by adding a
   `### Yanked` subsection with an explanation.
3. For Docker: you can re-tag the same image name at the fixed commit
   (re-push `:latest` and `:X.Y.Z`). The previous image ID is still
   reachable by digest if anyone needs it.

We don't delete git tags from the public repo — it breaks anyone's
pinned reference and is irrecoverable with hatch-vcs.
