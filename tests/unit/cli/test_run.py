"""Unit tests for the real `vkit run` command."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from voxkitchen.cli.main import app
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance


def _seed_manifest(path: Path, n: int = 3) -> None:
    cuts = [
        Cut(
            id=f"c{i}",
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
        for i in range(n)
    ]
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="fixture",
        stage_name="00_ingest",
    )
    CutSet(cuts).to_jsonl_gz(path, header)


def test_run_completes_simple_pipeline(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _seed_manifest(input_manifest)
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: clirun
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args:
    path: {input_manifest}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path)])
    assert result.exit_code == 0
    assert (work_dir / "00_s0" / "_SUCCESS").exists()


def test_run_with_stop_at(tmp_path: Path) -> None:
    input_manifest = tmp_path / "in.jsonl.gz"
    _seed_manifest(input_manifest)
    work_dir = tmp_path / "work"
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: stopat
work_dir: {work_dir}
num_cpu_workers: 1
ingest:
  source: manifest
  args: {{ path: {input_manifest} }}
stages:
  - {{ name: s0, op: identity }}
  - {{ name: s1, op: identity }}
  - {{ name: s2, op: identity }}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--stop-at", "s1"])
    assert result.exit_code == 0
    assert (work_dir / "01_s1" / "_SUCCESS").exists()
    assert not (work_dir / "02_s2").exists()
