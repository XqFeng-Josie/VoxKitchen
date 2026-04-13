"""Unit tests for voxkitchen.pipeline.runner.run_pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxkitchen.pipeline.loader import load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord, read_cuts
from voxkitchen.schema.provenance import Provenance


def _cut(cid: str) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec-1",
        start=0.0,
        duration=1.0,
        supervisions=[],
        provenance=Provenance(
            source_cut_id=None,
            generated_by="fixture",
            stage_name="00_ingest",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="fixture",
        ),
    )


def _write_input_manifest(path: Path, n: int = 4) -> None:
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture",
        stage_name="00_ingest",
    )
    CutSet([_cut(f"c{i}") for i in range(n)]).to_jsonl_gz(path, header)


def _write_pipeline_yaml(
    path: Path, work_dir: Path, input_manifest: Path, num_stages: int = 2
) -> None:
    stages = "\n".join(f"  - {{ name: s{i}, op: identity }}" for i in range(num_stages))
    path.write_text(
        f"""
version: "0.1"
name: runner-test
work_dir: {work_dir}
num_gpus: 1
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
{stages}
""",
        encoding="utf-8",
    )


def test_runner_completes_simple_pipeline(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=2)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    # Both stages should have written _SUCCESS + cuts.jsonl.gz
    assert (work_dir / "00_s0" / "_SUCCESS").exists()
    assert (work_dir / "00_s0" / "cuts.jsonl.gz").exists()
    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert (work_dir / "01_s1" / "cuts.jsonl.gz").exists()


def test_runner_preserves_cut_count_through_stages(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest, n=5)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    final_cuts = list(read_cuts(work_dir / "02_s2" / "cuts.jsonl.gz"))
    assert len(final_cuts) == 5


def test_runner_writes_run_yaml_snapshot(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    run_snapshot = work_dir / "run.yaml"
    assert run_snapshot.exists()
    text = run_snapshot.read_text()
    assert "runner-test" in text
    assert "run-fixed" in text


def test_runner_resume_skips_completed_stages(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest, n=3)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    # First run: complete all stages
    run_pipeline(spec)

    # Delete stage 2's output and re-run — stages 0 and 1 should be skipped
    (work_dir / "02_s2" / "_SUCCESS").unlink()
    (work_dir / "02_s2" / "cuts.jsonl.gz").unlink()

    # Touch stage 1's cuts to ensure it's NOT re-read (we want to verify skip)
    stage1_marker = work_dir / "01_s1" / "cuts.jsonl.gz"
    original_mtime = stage1_marker.stat().st_mtime

    run_pipeline(spec)

    # Stage 1 manifest must not have been rewritten
    assert stage1_marker.stat().st_mtime == original_mtime
    # Stage 2 must now be complete
    assert (work_dir / "02_s2" / "_SUCCESS").exists()


def test_runner_stops_at_stage(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=3)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec, stop_at="s1")

    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert not (work_dir / "02_s2").exists()


def test_runner_generates_report_on_success(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    _write_pipeline_yaml(pipeline_yaml, work_dir, input_manifest, num_stages=2)

    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")
    run_pipeline(spec)

    assert (work_dir / "report.html").exists()
    html = (work_dir / "report.html").read_text()
    assert "VoxKitchen" in html


def test_runner_raises_stage_failed_when_operator_missing(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _write_input_manifest(input_manifest)
    work_dir = tmp_path / "work"
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(
        f"""
version: "0.1"
name: bad-op
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
  - {{ name: s0, op: nonexistent_operator }}
""",
        encoding="utf-8",
    )
    spec = load_pipeline_spec(pipeline_yaml, run_id="run-fixed")

    with pytest.raises(StageFailedError):
        run_pipeline(spec)
