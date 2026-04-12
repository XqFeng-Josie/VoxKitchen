"""Unit tests for voxkitchen.pipeline.loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_minimal_pipeline(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest:
  source: manifest
  args:
    path: /tmp/cuts.jsonl.gz
stages:
  - name: one
    op: identity
""",
    )
    spec = load_pipeline_spec(path)
    assert spec.name == "demo"
    assert spec.ingest.source == "manifest"
    assert len(spec.stages) == 1
    assert spec.stages[0].name == "one"


def test_expands_name_and_run_id_in_work_dir(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/${name}-${run_id}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-FIXED")
    assert spec.work_dir == "/tmp/demo-run-FIXED"


def test_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/data/librispeech")
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: ${env:DATA_ROOT}/work
ingest: { source: manifest, args: { path: "${env:DATA_ROOT}/cuts.jsonl.gz" } }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-x")
    assert spec.work_dir == "/data/librispeech/work"
    assert spec.ingest.args["path"] == "/data/librispeech/cuts.jsonl.gz"


def test_unresolved_env_var_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: ${env:NOPE_NEVER_SET}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    with pytest.raises(PipelineLoadError, match="NOPE_NEVER_SET"):
        load_pipeline_spec(path, run_id="run-x")


def test_interpolation_does_not_touch_non_string_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
num_gpus: 4
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity, args: { threshold: 0.5 } }
""",
    )
    spec = load_pipeline_spec(path, run_id="run-x")
    assert spec.num_gpus == 4
    assert spec.stages[0].args["threshold"] == 0.5


def test_invalid_yaml_raises_load_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "version: '0.1'\nname: [this is: broken")
    with pytest.raises(PipelineLoadError):
        load_pipeline_spec(path, run_id="run-x")


def test_schema_validation_errors_wrapped_in_load_error(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/work
ingest: { source: smoke_signal, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    with pytest.raises(PipelineLoadError, match="source"):
        load_pipeline_spec(path, run_id="run-x")


def test_load_generates_run_id_when_not_provided(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
version: "0.1"
name: demo
work_dir: /tmp/${run_id}
ingest: { source: manifest, args: {} }
stages:
  - { name: s, op: identity }
""",
    )
    spec = load_pipeline_spec(path)
    assert spec.work_dir.startswith("/tmp/run-")
