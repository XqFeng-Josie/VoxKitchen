"""Unit tests for the real `vkit validate` command."""

from __future__ import annotations

import json
from pathlib import Path

import voxkitchen.cli.validate as validate_module
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
    assert "recommended image: slim" in result.output
    assert "vkit docker pull --tag slim" in result.output


def test_validate_recommends_fish_speech_image(tmp_path: Path) -> None:
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: fish
work_dir: /tmp/work
ingest: { source: manifest, args: { path: /tmp/in.jsonl.gz } }
stages:
  - { name: fish, op: tts_fish_speech }
""",
    )
    runner = CliRunner()
    # tts_fish_speech requires supervisions.text but no upstream provides it; pass
    # --no-preflight to isolate the image-recommendation check from field contracts.
    result = runner.invoke(app, ["validate", str(yaml_path), "--no-preflight"])
    assert result.exit_code == 0, result.output
    assert "recommended image: fish-speech" in result.output
    assert "vkit docker pull --tag fish-speech" in result.output


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


def test_validate_stage_args_hints_missing_operator_dependency(monkeypatch) -> None:
    monkeypatch.setattr(
        validate_module,
        "_validate_args_via_registry",
        lambda op_name, args: "__not_registered__",
    )
    monkeypatch.setattr(validate_module, "_operator_info_via_registry", lambda op_name: None)

    result = validate_module.validate_stage_args("tts_kokoro", {}, schemas=None)

    assert result.error is not None
    assert "vkit docker run --tag tts" in result.error


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


def test_validate_uses_schema_fallback_for_out_of_env_operator(tmp_path: Path, monkeypatch) -> None:
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
    assert "recommended image: asr" in result.output
    assert "run:  vkit docker run --tag asr" in result.output


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


# ---------------------------------------------------------------------------
# Pydantic error humanisation — the multi-line ``errors.pydantic.dev`` dump
# is replaced with a short bullet list, and ``extra_forbidden`` rows get a
# ``did you mean`` suggestion via ``difflib`` when one of the operator's
# known fields is close enough.
# ---------------------------------------------------------------------------


def test_validate_pydantic_errors_bulletised(tmp_path: Path) -> None:
    """Missing-required + unknown-arg shows two clean bullets; no pydantic URLs."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - name: q
    op: quality_score_filter
    args:
      filter_name: foo
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1, result.output
    # Bulletised format.
    assert "missing required arg: conditions" in result.output
    assert "unknown arg: filter_name" in result.output
    # And the noise is gone.
    assert "errors.pydantic.dev" not in result.output
    assert "validation errors for" not in result.output  # no "2 validation errors for X"


def test_validate_extra_forbidden_suggests_close_match(tmp_path: Path) -> None:
    """A typo (``target_channel``) for an existing field (``target_channels``)
    surfaces a ``did you mean`` hint from stdlib difflib."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - name: r
    op: resample
    args:
      target_sr: 16000
      target_channel: 1
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1, result.output
    assert "unknown arg: target_channel" in result.output
    assert "did you mean: target_channels?" in result.output


def test_validate_extra_forbidden_no_suggestion_when_no_close_match(tmp_path: Path) -> None:
    """If the typo isn't close to any allowed field we don't make one up."""
    yaml_path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: dir, args: { root: /tmp } }
stages:
  - name: r
    op: resample
    args:
      target_sr: 16000
      zzzzzzzz: nope
""",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(yaml_path)])
    assert result.exit_code == 1, result.output
    assert "unknown arg: zzzzzzzz" in result.output
    assert "did you mean" not in result.output
