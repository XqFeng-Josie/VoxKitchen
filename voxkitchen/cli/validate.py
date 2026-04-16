"""Real implementation of `vkit validate`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.operators.registry import UnknownOperatorError, get_operator
from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec


def validate_command(pipeline: Path) -> None:
    """Validate a pipeline YAML without executing it."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    errors: list[str] = []
    for stage in spec.stages:
        try:
            op_cls = get_operator(stage.op)
        except UnknownOperatorError as exc:
            errors.append(f"stage {stage.name!r}: {exc}")
            continue
        try:
            op_cls.config_cls.model_validate(stage.args)
        except Exception as exc:
            errors.append(f"stage {stage.name!r}: invalid args — {exc}")

    if errors:
        for e in errors:
            rprint(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1)

    rprint(
        f"[green]valid[/green]: {spec.name} "
        f"({len(spec.stages)} stage(s), ingest={spec.ingest.source})"
    )
