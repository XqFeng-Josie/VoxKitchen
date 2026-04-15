"""Real implementation of `vkit run`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

from voxkitchen.pipeline.loader import PipelineLoadError, load_pipeline_spec
from voxkitchen.pipeline.runner import StageFailedError, run_pipeline
from voxkitchen.pipeline.spec import PipelineSpec


def _print_dry_run(spec: PipelineSpec) -> bool:
    """Validate the pipeline and print a stage summary. Returns True if valid."""
    from voxkitchen.operators.registry import get_operator

    rprint(f"\n[bold]{spec.name}[/bold]  (dry-run)")
    rprint(f"  work_dir: {spec.work_dir}")
    rprint(f"  gc_mode:  {spec.gc_mode}")
    rprint(f"  gpus: {spec.num_gpus}  cpu_workers: {spec.num_cpu_workers or 'auto'}")
    rprint(f"  ingest:   source={spec.ingest.source}", end="")
    if spec.ingest.recipe:
        rprint(f"  recipe={spec.ingest.recipe}", end="")
    rprint()

    errors: list[str] = []
    t = Table(title="Stages")
    t.add_column("#", justify="right")
    t.add_column("Name")
    t.add_column("Operator")
    t.add_column("Device")
    t.add_column("Args")
    for i, stage in enumerate(spec.stages):
        try:
            op_cls = get_operator(stage.op)
            device: str = op_cls.device
            # Validate args against the operator's config schema
            op_cls.config_cls.model_validate(stage.args)
        except Exception as exc:
            device = "?"
            errors.append(f"stage {i} ({stage.name}): {exc}")
        args_str = ", ".join(f"{k}={v}" for k, v in stage.args.items()) if stage.args else "-"
        t.add_row(str(i), stage.name, stage.op, device, args_str)
    rprint(t)

    if errors:
        for err in errors:
            rprint(f"  [red]error:[/red] {err}")
        rprint(f"\n[red]validation failed ({len(errors)} error(s))[/red]")
        return False
    rprint("[green]validation passed[/green]")
    return True


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
    import logging

    logging.basicConfig(
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

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
        valid = _print_dry_run(spec)
        raise typer.Exit(code=0 if valid else 1)

    try:
        run_pipeline(
            spec,
            stop_at=stop_at,
            resume_from=resume_from,
            keep_intermediates=keep_intermediates,
        )
    except StageFailedError as exc:
        rprint(f"[red]stage failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    rprint("[green]pipeline complete[/green]")
