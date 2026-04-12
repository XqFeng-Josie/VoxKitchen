"""Real implementation of `vkit run`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline


def run_command(
    pipeline: Path,
    num_gpus: int | None = None,
    num_workers: int | None = None,
    work_dir: str | None = None,
    resume_from: str | None = None,
    stop_at: str | None = None,
    dry_run: bool = False,
    keep_intermediates: bool = False,
) -> None:
    """Execute a pipeline."""
    try:
        spec = load_pipeline_spec(pipeline)
    except PipelineLoadError as exc:
        rprint(f"[red]error loading pipeline:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # CLI flag overrides
    if num_gpus is not None:
        spec = spec.model_copy(update={"num_gpus": num_gpus})
    if num_workers is not None:
        spec = spec.model_copy(update={"num_cpu_workers": num_workers})
    if work_dir is not None:
        spec = spec.model_copy(update={"work_dir": work_dir})

    if dry_run:
        rprint("[yellow]--dry-run[/yellow] not yet implemented; Task 15 or later.")
        raise typer.Exit(code=0)

    try:
        run_pipeline(spec, stop_at=stop_at, keep_intermediates=keep_intermediates)
    except StageFailedError as exc:
        rprint(f"[red]stage failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    rprint("[green]pipeline complete[/green]")
