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

from pathlib import Path
from typing import Any

import typer
from rich import print as rprint

from voxkitchen.operators.registry import UnknownOperatorError, get_operator
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.runtime.schemas import load_op_schemas


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


def validate_command(pipeline: Path) -> None:
    """Validate a pipeline YAML without executing it."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    schemas = load_op_schemas()
    errors: list[str] = []

    for stage in spec.stages:
        err = _validate_args_via_registry(stage.op, stage.args)
        if err is None:
            continue
        if err != "__not_registered__":
            errors.append(f"stage {stage.name!r}: {err}")
            continue
        # Fallback path
        if schemas is None:
            errors.append(
                f"stage {stage.name!r}: unknown operator {stage.op!r} "
                "(no op_schemas.json available)"
            )
            continue
        fallback_err = _validate_args_via_json_schema(stage.op, stage.args, schemas)
        if fallback_err is not None:
            errors.append(f"stage {stage.name!r}: {fallback_err}")

    if errors:
        for e in errors:
            rprint(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)

    mode = "schema" if schemas is not None else "registry"
    rprint(
        f"[green]valid[/green]: {spec.name} "
        f"({len(spec.stages)} stage(s), ingest={spec.ingest.source}, validator={mode})"
    )
