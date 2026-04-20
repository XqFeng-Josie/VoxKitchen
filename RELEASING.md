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
- All tests must pass: `pytest -m "not slow and not gpu"`.
- `CHANGELOG.md` must have a section matching the new version (see below).

### 2. CHANGELOG

Move entries from `[Unreleased]` into a new dated section. Example:

```markdown
## [Unreleased]

## [0.2.0] — 2026-06-15

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

### 5. Docker images

`scripts/release.sh` builds and pushes each of the six targets in
smallest-to-largest order (so auth / disk / network failures surface
early on a cheap target) and tags each with both the rolling
`:<target>` and the pinned `:<target>-<version>`. See
[`scripts/release.sh`](scripts/release.sh) for the exact loop.

**Prerequisites**:

- Logged in to GHCR: `echo $GH_PAT | docker login ghcr.io -u XqFeng-Josie --password-stdin`.
  The PAT needs `write:packages` scope (classic PAT from
  https://github.com/settings/tokens/new).
- HF_TOKEN in `./.env` if you want to bake pyannote into the `asr` and
  `diarize` images (pipeline works without it, but first run will
  download ~80 MB of pyannote weights).

Build+push of all six targets on a well-provisioned machine takes
~1-2 hours (latest is the bottleneck at ~100 GB).

### 6. GitHub Release

The git tag is not the same as a public Release on GitHub. After the
tag is pushed, create the Release page to attach release notes:

```bash
# CLI (requires `gh` installed and authenticated)
gh release create v0.2.0 \
    --title "v0.2.0" \
    --notes-from-tag
```

Or via the web UI:
https://github.com/XqFeng-Josie/VoxKitchen/releases/new?tag=v0.2.0

Paste the relevant CHANGELOG section into the "Description" field and
publish.

### 7. Verify

One last sanity check against the published artifact:

```bash
# Pull a published image and run doctor against it
docker pull ghcr.io/xqfeng-josie/voxkitchen:slim
docker run --rm ghcr.io/xqfeng-josie/voxkitchen:slim doctor --expect core

# Expected: 36 operators available in core env, 0 missing.
```

## Why no CI for Docker publish

GitHub Actions default runners have ~14 GB of free disk. Our `:asr`,
`:tts`, and `:latest` images are 40-100 GB each and will not fit. Full
Docker publish CI requires either self-hosted runners or GitHub's
larger-runner add-ons.

Non-Docker CI (lint, unit tests, typecheck) runs on every push via
`.github/workflows/ci.yml` — that's the frequent regression gate.

## Yanking a release

If a pushed release has a critical bug:

1. **Publish a patch release** with the fix (`vX.Y.Z+1`) — this is the
   normal path and what pip users expect.
2. **Mark the broken release as such** in `CHANGELOG.md` by adding a
   `### Yanked` subsection with an explanation.
3. For Docker: you can re-tag the same image name at the fixed commit
   (re-push `:latest` and `:X.Y.Z`). The previous image ID is still
   reachable by digest if anyone needs it.

We don't delete git tags from the public repo — it breaks anyone's
pinned reference and is irrecoverable with hatch-vcs.
