"""End-to-end integration test for the Plan 2 pipeline engine.

Uses a real YAML, a real manifest on disk, the real runner, and the
IdentityOperator across multiple stages. Verifies the full cycle:
load → ingest → stage 0 → stage 1 → stage 2 → pack as manifest.
"""

from __future__ import annotations

from pathlib import Path

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import run_pipeline
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, read_cuts, read_header


def test_end_to_end_pipeline_preserves_cuts(
    tmp_path: Path, sample_cutset: CutSet, sample_manifest_path: Path
) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: e2e
work_dir: {work_dir}
num_cpu_workers: 2
ingest:
  source: manifest
  args:
    path: {sample_manifest_path}
stages:
  - {{ name: pre, op: identity }}
  - {{ name: mid, op: identity }}
  - {{ name: post, op: identity }}
""",
        encoding="utf-8",
    )

    spec = load_pipeline_spec(yaml_path, run_id="run-e2e")
    run_pipeline(spec)

    # All three stages completed
    for i, name in enumerate(["pre", "mid", "post"]):
        stage_dir = work_dir / f"{i:02d}_{name}"
        assert stage_dir.exists()
        assert (stage_dir / "_SUCCESS").exists()
        assert (stage_dir / "cuts.jsonl.gz").exists()

    # Final output cuts match input
    final = list(read_cuts(work_dir / "02_post" / "cuts.jsonl.gz"))
    assert sorted(c.id for c in final) == sorted(c.id for c in sample_cutset)

    # run.yaml snapshot exists
    assert (work_dir / "run.yaml").exists()

    # Each stage's header has the right stage_name
    for i, name in enumerate(["pre", "mid", "post"]):
        h = read_header(work_dir / f"{i:02d}_{name}" / "cuts.jsonl.gz")
        assert h.stage_name == name
        assert h.schema_version == SCHEMA_VERSION


def test_end_to_end_resume_after_deleted_final_stage(
    tmp_path: Path, sample_manifest_path: Path
) -> None:
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: e2e-resume
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args: {{ path: {sample_manifest_path} }}
stages:
  - {{ name: a, op: identity }}
  - {{ name: b, op: identity }}
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(yaml_path, run_id="run-resume")

    # First run
    run_pipeline(spec)
    assert (work_dir / "01_b" / "_SUCCESS").exists()

    # Simulate partial failure: delete final stage's manifest and success marker
    (work_dir / "01_b" / "_SUCCESS").unlink()
    (work_dir / "01_b" / "cuts.jsonl.gz").unlink()

    # Second run should pick up from stage b
    run_pipeline(spec)
    assert (work_dir / "01_b" / "_SUCCESS").exists()
    assert (work_dir / "01_b" / "cuts.jsonl.gz").exists()
