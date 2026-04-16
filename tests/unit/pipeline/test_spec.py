"""Unit tests for voxkitchen.pipeline.spec."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from voxkitchen.pipeline.spec import IngestSpec, PipelineSpec, StageSpec


def _ingest() -> IngestSpec:
    return IngestSpec(source="manifest", args={"path": "/tmp/in.jsonl.gz"})


def _stages(*names: str) -> list[StageSpec]:
    return [StageSpec(name=n, op="identity", args={}) for n in names]


def test_stage_spec_construction() -> None:
    s = StageSpec(name="vad", op="silero_vad", args={"threshold": 0.5})
    assert s.name == "vad"
    assert s.op == "silero_vad"
    assert s.args == {"threshold": 0.5}


def test_stage_spec_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        StageSpec.model_validate({"name": "x", "op": "y", "args": {}, "extra": "boom"})


def test_ingest_spec_requires_known_source() -> None:
    with pytest.raises(ValidationError):
        IngestSpec.model_validate({"source": "smoke_signal", "args": {}})


def test_pipeline_spec_minimal() -> None:
    spec = PipelineSpec(
        version="0.1",
        name="demo",
        work_dir="/tmp/work",
        ingest=_ingest(),
        stages=_stages("one", "two"),
    )
    assert spec.version == "0.1"
    assert spec.num_gpus == 1
    assert spec.gc_mode == "aggressive"
    assert spec.description == ""
    assert spec.num_cpu_workers is None


def test_pipeline_spec_rejects_duplicate_stage_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        PipelineSpec(
            version="0.1",
            name="demo",
            work_dir="/tmp/work",
            ingest=_ingest(),
            stages=[
                StageSpec(name="dup", op="identity"),
                StageSpec(name="dup", op="identity"),
            ],
        )


def test_pipeline_spec_rejects_invalid_gc_mode() -> None:
    with pytest.raises(ValidationError):
        PipelineSpec.model_validate(
            {
                "version": "0.1",
                "name": "demo",
                "work_dir": "/tmp/work",
                "gc_mode": "paranoid",
                "ingest": {"source": "manifest", "args": {}},
                "stages": [],
            }
        )


def test_pipeline_spec_requires_at_least_one_stage() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        PipelineSpec(
            version="0.1",
            name="demo",
            work_dir="/tmp/work",
            ingest=_ingest(),
            stages=[],
        )


def test_pipeline_spec_forbids_extra_top_level_fields() -> None:
    with pytest.raises(ValidationError):
        PipelineSpec.model_validate(
            {
                "version": "0.1",
                "name": "demo",
                "work_dir": "/tmp/work",
                "ingest": {"source": "manifest", "args": {}},
                "stages": [{"name": "s", "op": "identity", "args": {}}],
                "future_field": "boom",
            }
        )
