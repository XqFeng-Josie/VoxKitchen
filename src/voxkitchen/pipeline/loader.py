"""YAML → PipelineSpec loader with string interpolation.

Supported placeholders inside any string value:

- ``${name}`` — the top-level pipeline ``name``
- ``${run_id}`` — the pipeline run id (generated or provided by the caller)
- ``${env:VAR_NAME}`` — value of the environment variable ``VAR_NAME``

Placeholders are resolved by a single recursive pass over the parsed YAML.
Non-string values pass through unchanged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from voxkitchen.pipeline.spec import PipelineSpec
from voxkitchen.utils.run_id import generate_run_id

_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


class PipelineLoadError(ValueError):
    """Raised when a pipeline YAML cannot be parsed, interpolated, or validated."""


def load_pipeline_spec(path: Path, run_id: str | None = None) -> PipelineSpec:
    """Load a pipeline YAML file and return a validated ``PipelineSpec``.

    ``run_id`` is used to expand ``${run_id}`` placeholders. If omitted, a
    fresh id is generated via ``generate_run_id()``.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineLoadError(f"cannot read {path}: {exc}") from exc

    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise PipelineLoadError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineLoadError(
            f"pipeline YAML must be a mapping at the top level, got {type(raw).__name__}"
        )

    effective_run_id = run_id if run_id is not None else generate_run_id()
    pipeline_name = raw.get("name", "")

    interpolated = _interpolate(raw, name=str(pipeline_name), run_id=effective_run_id)
    # Inject the effective run_id so the runner can retrieve it without re-parsing paths.
    interpolated["run_id"] = effective_run_id

    try:
        return PipelineSpec.model_validate(interpolated)
    except ValidationError as exc:
        raise PipelineLoadError(f"pipeline validation failed for {path}:\n{exc}") from exc


def _interpolate(obj: Any, *, name: str, run_id: str) -> Any:
    """Recursively replace ``${...}`` placeholders in string values."""
    if isinstance(obj, str):
        return _interpolate_string(obj, name=name, run_id=run_id)
    if isinstance(obj, dict):
        return {k: _interpolate(v, name=name, run_id=run_id) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(v, name=name, run_id=run_id) for v in obj]
    return obj


def _interpolate_string(s: str, *, name: str, run_id: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "name":
            return name
        if key == "run_id":
            return run_id
        if key.startswith("env:"):
            env_var = key[4:]
            value = os.environ.get(env_var)
            if value is None:
                raise PipelineLoadError(
                    f"environment variable {env_var!r} referenced by ${{env:{env_var}}} is not set"
                )
            return value
        raise PipelineLoadError(f"unknown placeholder: ${{{key}}}")

    return _INTERP_RE.sub(replace, s)
