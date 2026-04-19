"""Unit tests for the real `vkit validate` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner
from voxkitchen.cli.main import app
from voxkitchen.runtime import schemas as schemas_module


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_validate_accepts_valid_pipeline(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: s0, op: identity }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_rejects_unknown_operator(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: s0, op: not_a_real_operator }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1
    assert "not_a_real_operator" in result.output


def test_validate_rejects_malformed_yaml(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path, "version: 0.1\nstages: [bad: syntax")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1


def _write_fake_schemas(tmp_path: Path) -> Path:
    """Simulate an op_schemas.json exported by another env.

    ``fake_remote_op`` takes a required int ``threshold`` and an optional
    string ``mode``. It is *not* registered in this process, so validate
    must fall back to JSON-schema checking.
    """
    schemas_path = tmp_path / "op_schemas.json"
    schemas_path.write_text(
        json.dumps(
            {
                "fake_remote_op": {
                    "config_schema": {
                        "type": "object",
                        "properties": {
                            "threshold": {"type": "integer"},
                            "mode": {"type": "string"},
                        },
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


def test_validate_uses_schema_fallback_for_out_of_env_operator(
    tmp_path: Path, monkeypatch
) -> None:
    schemas_path = _write_fake_schemas(tmp_path)
    monkeypatch.setenv("VKIT_OP_SCHEMAS", str(schemas_path))
    schemas_module.reset_cache()

    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: s0, op: fake_remote_op, args: { threshold: 5, mode: strict } }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 0, result.output
    assert "validator=schema" in result.output


def test_validate_schema_fallback_catches_bad_args(tmp_path: Path, monkeypatch) -> None:
    schemas_path = _write_fake_schemas(tmp_path)
    monkeypatch.setenv("VKIT_OP_SCHEMAS", str(schemas_path))
    schemas_module.reset_cache()

    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: bad_types, op: fake_remote_op, args: { threshold: "not-an-int" } }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1
    assert "bad_types" in result.output
    assert "invalid args" in result.output


def test_validate_schema_fallback_catches_missing_required(tmp_path: Path, monkeypatch) -> None:
    schemas_path = _write_fake_schemas(tmp_path)
    monkeypatch.setenv("VKIT_OP_SCHEMAS", str(schemas_path))
    schemas_module.reset_cache()

    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: missing_threshold, op: fake_remote_op, args: { mode: strict } }
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1
    assert "missing_threshold" in result.output
