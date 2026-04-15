"""Top-level Typer application exposing the `vkit` CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.cli.inspect import inspect_app
from voxkitchen.cli.operators_cmd import operators_app

app = typer.Typer(
    name="vkit",
    help="VoxKitchen — declarative speech data processing toolkit.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(inspect_app, name="inspect")
app.add_typer(operators_app, name="operators")


@app.command(help="Scaffold a new pipeline project directory.")
def init(path: Path = typer.Argument(..., help="Target directory.")) -> None:
    from voxkitchen.cli.init_cmd import init_project

    try:
        init_project(path)
    except FileExistsError as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    rprint(f"[green]created[/green] pipeline project at {path}")


@app.command(help="Build an initial CutSet from a data source.")
def ingest(
    source: str = typer.Option(..., "--source", help="dir | manifest | recipe"),
    out: Path = typer.Option(..., "--out", help="Output cuts.jsonl.gz path"),
    root: str | None = typer.Option(None, "--root", help="Root directory (for dir/recipe)"),
    path: str | None = typer.Option(None, "--path", help="Manifest path (for source=manifest)"),
    recipe: str | None = typer.Option(None, "--recipe", help="Recipe name"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
    subsets: str | None = typer.Option(None, "--subsets", help="Comma-separated subset names"),
) -> None:
    from voxkitchen.cli.ingest_cmd import ingest_command

    ingest_command(
        source=source,
        out=out,
        root=root,
        path=path,
        recipe=recipe,
        recursive=recursive,
        subsets=subsets,
    )


@app.command(help="Parse and validate a pipeline YAML (no execution).")
def validate(
    pipeline: Path = typer.Argument(..., help="Pipeline YAML path."),
) -> None:
    from voxkitchen.cli.validate import validate_command

    validate_command(pipeline)


@app.command(help="Execute a pipeline.")
def run(
    pipeline: Path = typer.Argument(..., help="Pipeline YAML path."),
    num_gpus: int | None = typer.Option(None, "--num-gpus", help="Override num_gpus."),
    num_workers: int | None = typer.Option(None, "--num-workers", help="Override num_cpu_workers."),
    work_dir: str | None = typer.Option(None, "--work-dir", help="Override work_dir."),
    resume_from: str | None = typer.Option(
        None, "--resume-from", help="Stage name to resume from."
    ),
    stop_at: str | None = typer.Option(None, "--stop-at", help="Stop after this stage."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, do not execute."),
    keep_intermediates: bool = typer.Option(
        False, "--keep-intermediates", help="Disable GC; keep all derived files."
    ),
) -> None:
    from voxkitchen.cli.run import run_command

    run_command(
        pipeline=pipeline,
        num_gpus=num_gpus,
        num_workers=num_workers,
        work_dir=work_dir,
        resume_from=resume_from,
        stop_at=stop_at,
        dry_run=dry_run,
        keep_intermediates=keep_intermediates,
    )


@app.command(help="Launch local Gradio panel to explore a CutSet.")
def viz(
    path: Path = typer.Argument(..., help="CutSet manifest path."),
    port: int = typer.Option(7860, "--port", help="Port for local server."),
) -> None:
    try:
        from voxkitchen.viz.panel.app import launch

        launch(str(path), port=port)
    except ImportError:
        rprint("[red]error:[/red] Gradio not installed. Run: pip install voxkitchen\\[viz-panel]")
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
