"""Implementation of ``vkit validate``.

Validates a pipeline YAML *without* importing every operator the pipeline
references. In multi-env images the parent (core) env cannot import ASR
or TTS operator classes — their deps live in other envs — so we fall
back to JSON-schema validation against the merged ``op_schemas.json``.

Fast path (operator importable here): Pydantic ``model_validate``.
Fallback (operator lives elsewhere): ``jsonschema`` against its config
schema exported at image-build time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from voxkitchen.cli.hints import (
    format_missing_operator_hint,
    format_recommended_image_hint,
    recommend_docker_tag,
)
from voxkitchen.operators.registry import UnknownOperatorError, get_operator
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.runtime.schemas import load_op_schemas


@dataclass(frozen=True)
class StageValidation:
    """Result of validating one pipeline stage's operator args."""

    error: str | None
    validator: str
    device: str
    required_extras: list[str]


def _validate_args_via_registry(op_name: str, args: dict[str, Any]) -> str | None:
    """Try the fast path. Returns an error message, ``"__not_registered__"``
    if the op is not importable here, or ``None`` on success."""
    try:
        op_cls = get_operator(op_name)
    except UnknownOperatorError:
        return "__not_registered__"
    try:
        op_cls.config_cls.model_validate(args)
    except Exception as exc:
        return f"invalid args — {exc}"
    return None


def _operator_info_via_registry(op_name: str) -> tuple[str, list[str]] | None:
    """Return ``(device, required_extras)`` when the operator is importable here."""
    try:
        op_cls = get_operator(op_name)
    except UnknownOperatorError:
        return None
    return (str(op_cls.device), list(op_cls.required_extras))


def _validate_args_via_json_schema(
    op_name: str,
    args: dict[str, Any],
    schemas: dict[str, dict[str, Any]],
) -> str | None:
    """Fallback for operators not importable in this env. Returns an error
    string, or ``None`` on success."""
    if op_name not in schemas:
        return f"unknown operator {op_name!r} (not in registry, not in op_schemas.json)"

    schema = schemas[op_name].get("config_schema")
    if not isinstance(schema, dict):
        # Should not happen — dump_schemas always writes an object, even
        # on error. Treat as unknown so the user sees a clear message.
        return f"no config schema recorded for {op_name!r}"

    # Import jsonschema lazily: it's in core deps, but lazy keeps CLI startup snappy.
    try:
        import jsonschema
    except ImportError:
        return (
            f"cannot validate {op_name!r}: operator lives in another env "
            "and jsonschema is not installed in this env"
        )

    try:
        jsonschema.validate(instance=args, schema=schema)
    except jsonschema.ValidationError as exc:
        # jsonschema error messages include the failing path + value; keep
        # only the first line so the CLI output stays readable.
        first = str(exc.message).split("\n", 1)[0]
        return f"invalid args — {first}"
    return None


def validate_stage_args(
    op_name: str,
    args: dict[str, Any],
    schemas: dict[str, dict[str, Any]] | None = None,
) -> StageValidation:
    """Validate one operator config, with schema fallback for out-of-env ops."""
    err = _validate_args_via_registry(op_name, args)
    info = _operator_info_via_registry(op_name)
    if err is None:
        device, extras = info if info is not None else ("?", [])
        return StageValidation(None, "registry", device, extras)

    if err != "__not_registered__":
        device, extras = info if info is not None else ("?", [])
        return StageValidation(err, "registry", device, extras)

    if schemas is None:
        hint = format_missing_operator_hint(op_name)
        tail = f"; {hint}" if hint else ""
        return StageValidation(
            f"unknown operator {op_name!r} (no op_schemas.json available){tail}",
            "registry",
            "?",
            [],
        )

    schema_info = schemas.get(op_name)
    if schema_info is None:
        hint = format_missing_operator_hint(op_name)
        tail = f"; {hint}" if hint else ""
        return StageValidation(
            f"unknown operator {op_name!r} (not in registry, not in op_schemas.json){tail}",
            "schema",
            "?",
            [],
        )

    fallback_err = _validate_args_via_json_schema(op_name, args, schemas)
    return StageValidation(
        fallback_err,
        "schema",
        str(schema_info.get("device", "?")),
        [str(e) for e in schema_info.get("required_extras", [])],
    )


def validate_command(pipeline: Path) -> None:
    """Validate a pipeline YAML without executing it."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    schemas = load_op_schemas()
    errors: list[str] = []

    results: list[StageValidation] = []
    for stage in spec.stages:
        result = validate_stage_args(stage.op, stage.args, schemas)
        results.append(result)
        if result.error is not None:
            errors.append(f"stage {stage.name!r}: {result.error}")

    if errors:
        for e in errors:
            rprint(f"[red]error:[/red] {e}")
        rprint(
            "[dim]Tip: use `vkit docker run <yaml> --dry-run` to validate "
            "inside the selected image, or `vkit docker doctor` to check it.[/dim]"
        )
        raise typer.Exit(code=1)

    mode = "schema" if schemas is not None else "registry"
    rprint(
        f"[green]valid[/green]: {spec.name} "
        f"({len(spec.stages)} stage(s), ingest={spec.ingest.source}, validator={mode})"
    )
    tag = recommend_docker_tag([r.required_extras for r in results])
    rprint(f"[dim]{format_recommended_image_hint(tag, str(pipeline))}[/dim]")
