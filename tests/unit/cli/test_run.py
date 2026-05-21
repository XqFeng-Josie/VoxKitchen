"""Unit tests for the real `vkit run` command."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import voxkitchen.cli.run as run_module
from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec
from voxkitchen.runtime import schemas as schemas_module
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


def test_run_stage_failure_exits_1_not_2(tmp_path: Path, monkeypatch) -> None:
    """Stage-level failures are runtime errors (exit 1), not bad-invocation (exit 2).

    The convention across the CLI: code 1 covers "ran but failed" (file
    missing, validation failed, stage exception); code 2 is reserved for
    "invocation is malformed" (unknown flag, missing docker binary, unknown
    category). A stage exception belongs in the first bucket.
    """
    from voxkitchen.cli import run as run_cli
    from voxkitchen.pipeline.runner import StageFailedError

    def _raise_stage_failure(*_args, **_kwargs) -> None:
        raise StageFailedError("vad", RuntimeError("synthetic operator crash"))

    monkeypatch.setattr(run_cli, "run_pipeline", _raise_stage_failure)

    input_manifest = tmp_path / "in.jsonl.gz"
    _seed_manifest(input_manifest)
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: stagefail
work_dir: {tmp_path / "work"}
ingest: {{ source: manifest, args: {{ path: {input_manifest} }} }}
stages:
  - {{ name: vad, op: identity }}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["run", str(yaml_path)])
    assert result.exit_code == 1, result.output
    assert "stage failed" in result.output


def _write_fake_schemas(tmp_path: Path) -> Path:
    schemas_path = tmp_path / "op_schemas.json"
    schemas_path.write_text(
        json.dumps(
            {
                "fake_remote_op": {
                    "config_schema": {
                        "type": "object",
                        "properties": {"threshold": {"type": "integer"}},
                        "required": ["threshold"],
                        "additionalProperties": False,
                    },
                    "required_extras": ["asr"],
                    "device": "gpu",
                    "module": "fake",
                    "doc": "",
                }
            }
        ),
        encoding="utf-8",
    )
    return schemas_path


def test_run_warns_when_used_outside_managed_runtime(tmp_path: Path, monkeypatch) -> None:
    # The runtime-detection logic now lives in voxkitchen.cli.hints — patch
    # it there so warn_if_unmanaged_runtime() sees False regardless of host.
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: False)

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: dryrun
work_dir: {tmp_path / "work"}
ingest: {{ source: manifest, args: {{ path: {tmp_path / "in.jsonl.gz"} }} }}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["run", str(yaml_path), "--dry-run"])

    assert result.exit_code == 0
    assert "current Python environment" in result.output
    assert "VoxKitchen Docker runtimes" in result.output
    assert "vkit docker run <yaml>" in result.output


def test_run_does_not_warn_inside_managed_runtime(tmp_path: Path, monkeypatch) -> None:
    import voxkitchen.cli.hints as hints_module

    monkeypatch.setattr(hints_module, "is_managed_runtime", lambda: True)

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: dryrun
work_dir: {tmp_path / "work"}
ingest: {{ source: manifest, args: {{ path: {tmp_path / "in.jsonl.gz"} }} }}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["run", str(yaml_path), "--dry-run"])

    assert result.exit_code == 0
    assert "intended for VoxKitchen Docker runtimes" not in result.output


def test_run_dry_run_uses_schema_fallback_for_out_of_env_operator(
    tmp_path: Path, monkeypatch
) -> None:
    schemas_path = _write_fake_schemas(tmp_path)
    monkeypatch.setenv("VKIT_OP_SCHEMAS", str(schemas_path))
    schemas_module.reset_cache()

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        """
version: "0.1"
name: dryrun
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: remote, op: fake_remote_op, args: { threshold: 5 } }
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--dry-run"])
    schemas_module.reset_cache()

    assert result.exit_code == 0, result.output
    assert "fake_remote_op" in result.output
    assert "schema" in result.output
    assert "gpu" in result.output
    assert "recommended image: asr" in result.output


def test_run_dry_run_schema_fallback_reports_bad_args(tmp_path: Path, monkeypatch) -> None:
    schemas_path = _write_fake_schemas(tmp_path)
    monkeypatch.setenv("VKIT_OP_SCHEMAS", str(schemas_path))
    schemas_module.reset_cache()

    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        """
version: "0.1"
name: dryrun
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: remote, op: fake_remote_op, args: { threshold: "bad" } }
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--dry-run"])
    schemas_module.reset_cache()

    assert result.exit_code == 1
    assert "invalid args" in result.output
    assert "vkit docker doctor" in result.output


def test_run_dry_run_warns_when_ingest_dir_is_missing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "pipeline.yaml"
    missing = tmp_path / "missing-data"
    yaml_path.write_text(
        f"""
version: "0.1"
name: dryrun
work_dir: {tmp_path / "work"}
ingest:
  source: dir
  args:
    root: {missing}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--dry-run"])

    assert result.exit_code == 0
    assert "warning:" in result.output
    assert "ingest root" in result.output


def test_run_dry_run_warns_when_ingest_dir_has_no_audio(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text(
        f"""
version: "0.1"
name: dryrun
work_dir: {tmp_path / "work"}
ingest:
  source: dir
  args:
    root: {data_dir}
stages:
  - {{ name: s0, op: identity }}
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["run", str(yaml_path), "--dry-run"])

    assert result.exit_code == 0
    assert "contains no supported audio files" in " ".join(result.output.split())


def test_pack_huggingface_output_hint_uses_configured_output_dir(tmp_path: Path) -> None:
    spec = PipelineSpec(
        version="0.1",
        name="hf",
        work_dir=str(tmp_path / "work"),
        ingest=IngestSpec(source="manifest", args={"path": str(tmp_path / "in.jsonl.gz")}),
        stages=[
            StageSpec(name="prep", op="identity"),
            StageSpec(
                name="pack",
                op="pack_huggingface",
                args={"output_dir": "./output/hf_dataset"},
            ),
        ],
    )

    assert run_module._pack_huggingface_output_dir(spec, max_stage_idx=1) == Path(
        "./output/hf_dataset"
    )


def test_pack_huggingface_output_hint_uses_stage_default(tmp_path: Path) -> None:
    spec = PipelineSpec(
        version="0.1",
        name="hf",
        work_dir=str(tmp_path / "work"),
        ingest=IngestSpec(source="manifest", args={"path": str(tmp_path / "in.jsonl.gz")}),
        stages=[StageSpec(name="pack", op="pack_huggingface")],
    )

    assert run_module._pack_huggingface_output_dir(spec, max_stage_idx=0) == (
        tmp_path / "work" / "00_pack" / "hf_output"
    )
