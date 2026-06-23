"""Unit tests for `vkit schema` subcommands."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.cli.schema_cmd import SCHEMA_URI, build_pipeline_schema


def test_build_pipeline_schema_has_expected_top_level_keys() -> None:
    schema = build_pipeline_schema()
    assert schema["$schema"] == SCHEMA_URI
    assert schema["title"] == "VoxKitchen pipeline"
    assert "$defs" in schema
    assert "StageSpec" in schema["$defs"]


def test_build_pipeline_schema_constrains_op_to_registered_names() -> None:
    """`op` must be an enum of registered operators so editors can autocomplete."""
    from voxkitchen.operators.registry import list_operators

    schema = build_pipeline_schema()
    op_prop = schema["$defs"]["StageSpec"]["properties"]["op"]
    assert "enum" in op_prop
    assert set(op_prop["enum"]) == set(list_operators())
    # Sanity: a few well-known operators are there
    assert "resample" in op_prop["enum"]
    assert "pack_jsonl" in op_prop["enum"]


def test_schema_export_writes_file_at_specified_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "pipeline.schema.json"
    result = CliRunner().invoke(app, ["schema", "export", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()

    data = json.loads(out.read_text())
    assert data["title"] == "VoxKitchen pipeline"
    # File ends in newline (POSIX-friendly, plays well with git)
    assert out.read_text().endswith("\n")


def test_schema_export_default_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["schema", "export"])
    assert result.exit_code == 0
    assert (tmp_path / "pipeline.schema.json").exists()


def test_committed_pipeline_schema_matches_cli_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    schema_path = repo_root / "docs" / "schemas" / "pipeline.schema.json"
    generated_path = tmp_path / "pipeline.schema.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "voxkitchen.cli.main",
            "schema",
            "export",
            "--out",
            str(generated_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    committed = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = json.loads(generated_path.read_text(encoding="utf-8"))

    assert committed == generated, (
        "docs/schemas/pipeline.schema.json is stale; run "
        "`python -m voxkitchen.cli.main schema export --out "
        "docs/schemas/pipeline.schema.json` and commit the result."
    )


def test_schema_validates_a_minimal_pipeline() -> None:
    """The schema must accept the same shape PipelineSpec.model_validate accepts."""
    import jsonschema

    schema = build_pipeline_schema()
    validator = jsonschema.Draft202012Validator(schema)

    valid = {
        "version": "0.1",
        "name": "demo",
        "work_dir": "/tmp/work",
        "ingest": {"source": "manifest", "args": {"path": "x.jsonl.gz"}},
        "stages": [{"name": "s", "op": "resample"}],
    }
    assert list(validator.iter_errors(valid)) == []


def test_schema_rejects_unknown_operator_name() -> None:
    """A typo in `op:` must surface as a schema error so editors can flag it."""
    import jsonschema

    schema = build_pipeline_schema()
    validator = jsonschema.Draft202012Validator(schema)

    invalid = {
        "version": "0.1",
        "name": "demo",
        "work_dir": "/tmp/work",
        "ingest": {"source": "manifest", "args": {}},
        "stages": [{"name": "s", "op": "not_a_real_operator"}],
    }
    errors = list(validator.iter_errors(invalid))
    assert errors, "schema accepted an unknown operator name"
    assert any("not_a_real_operator" in str(e.message) for e in errors)
