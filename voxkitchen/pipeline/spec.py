"""Pydantic models for pipeline YAML.

These are the canonical in-memory representation of a pipeline. The YAML
loader in ``loader.py`` is responsible for turning raw ``dict`` into these
objects (including string interpolation); the runner consumes them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


class StageSpec(BaseModel):
    """One stage in a pipeline — a named invocation of an operator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    op: str
    args: dict[str, Any] = {}


class IngestSpec(BaseModel):
    """How Cuts enter the pipeline before any stage runs."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["dir", "manifest", "recipe"]
    args: dict[str, Any] = {}
    recipe: str | None = None


class PipelineSpec(BaseModel):
    """Top-level pipeline specification."""

    model_config = ConfigDict(extra="forbid")

    version: str
    name: str
    description: str = ""
    work_dir: str
    run_id: str | None = None  # set by loader after ${run_id} expansion
    num_gpus: int = 1
    num_cpu_workers: int | None = None
    gc_mode: Literal["aggressive", "keep"] = "aggressive"
    ingest: IngestSpec
    stages: list[StageSpec]

    @field_validator("stages")
    @classmethod
    def _stages_non_empty_and_unique(cls, v: list[StageSpec]) -> list[StageSpec]:
        if len(v) == 0:
            raise ValueError("pipeline must declare at least one stage")
        names = [s.name for s in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"stage names must be unique, duplicates: {sorted(set(duplicates))}")
        return v
