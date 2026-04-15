# Data Protocol

VoxKitchen uses a Lhotse-inspired data protocol with three core types.

## Recording

A **Recording** describes a physical audio file. It stores metadata (sample rate, duration, channels) and a reference to the file path.

## Supervision

A **Supervision** is a labeled time interval over a Recording. It carries optional annotations: `text` (transcription), `speaker`, `language`, `gender`.

Multiple Supervisions can overlap on the same Recording (e.g., two speakers talking simultaneously).

## Cut

A **Cut** is the unit that flows through a pipeline. It references a slice of a Recording plus all Supervisions within that slice.

Key fields:
- `id` — unique identifier
- `recording` — embedded Recording (optional, for audio access)
- `supervisions` — list of annotations
- `metrics` — computed values (e.g., SNR)
- `provenance` — where this Cut came from (parent Cut, which operator, which pipeline run)

## CutSet

A **CutSet** is a collection of Cuts, serialized as `cuts.jsonl.gz` (gzipped JSON lines with a header record).

## Provenance

Every Cut carries a `Provenance` record linking it to its parent Cut and the operator that produced it. This forms a DAG that can be traversed via `vkit inspect trace <cut_id> --in <work_dir>`.

## Serialization

Manifests use JSONL.gz format:
- Line 1: header with `schema_version` and `pipeline_run_id`
- Lines 2+: one Cut per line as JSON

Example:
```json
{"__type__": "voxkitchen.header", "schema_version": "0.1", ...}
{"__type__": "cut", "id": "utt-001", "duration": 3.5, ...}
```
