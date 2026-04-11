# VoxKitchen

A researcher-friendly, declarative speech data processing toolkit with a unified data protocol.

> **Status:** Pre-alpha. This repository is under active development and the API is unstable.

## What it does

VoxKitchen takes raw audio (local directories, manifests, or open-source datasets) and runs it through declarative YAML pipelines that segment, auto-label, quality-filter, and package it for training. Every intermediate step is inspectable, every run is resumable, and every output carries full provenance.

## Installation

```bash
pip install voxkitchen
```

## Quickstart

```bash
vkit --help
```

Full documentation coming soon. See `docs/superpowers/specs/` for the design spec.

## License

Apache 2.0. See [LICENSE](LICENSE).
