"""Top-level Typer application exposing the `vkit` CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.cli.datasets_cmd import datasets_app
from voxkitchen.cli.docker_cmd import docker_app
from voxkitchen.cli.doctor import doctor_app
from voxkitchen.cli.inspect import inspect_app
from voxkitchen.cli.operators_cmd import operators_app
from voxkitchen.cli.recipes_cmd import recipes_app
from voxkitchen.cli.schema_cmd import schema_app

app = typer.Typer(
    name="vkit",
    help="VoxKitchen — declarative speech data processing toolkit.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(inspect_app, name="inspect")
app.add_typer(operators_app, name="operators")
app.add_typer(recipes_app, name="recipes")
app.add_typer(datasets_app, name="datasets")
app.add_typer(doctor_app, name="doctor")
app.add_typer(docker_app, name="docker")
app.add_typer(schema_app, name="schema")


@app.command(help="Scaffold a new pipeline project directory.")
def init(
    path: Path = typer.Argument(None, help="Target directory."),
    template: str | None = typer.Option(
        None, "--template", "-t", help="Pipeline template (tts, asr, cleaning, speaker)."
    ),
    list_templates: bool = typer.Option(
        False, "--list-templates", help="List available templates and exit."
    ),
) -> None:
    if list_templates:
        from voxkitchen.cli.init_cmd import list_templates as _list

        _list()
        return

    if path is None:
        rprint("[red]error:[/red] please provide a target directory.")
        rprint("[dim]Usage: vkit init <path> [--template tts|asr|cleaning|speaker][/dim]")
        raise typer.Exit(code=1)

    from voxkitchen.cli.init_cmd import init_project, recommended_docker_tag

    try:
        init_project(path, template=template)
    except (FileExistsError, KeyError) as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    msg = f"[green]created[/green] pipeline project at {path}"
    if template:
        msg += f" (template: {template})"
    rprint(msg)
    rprint("\n[bold]Next steps[/bold]")
    rprint(f"  [dim]1.[/dim] cd {path}")
    rprint("  [dim]2.[/dim] cp /path/to/audio/* data/")
    rprint("  [dim]3.[/dim] vkit show pipeline.yaml         [dim]# preview the chain[/dim]")
    rprint("  [dim]4.[/dim] vkit validate pipeline.yaml     [dim]# catch typos[/dim]")
    tag = recommended_docker_tag(template)
    rprint(
        f"  [dim]5.[/dim] vkit docker run --tag {tag} pipeline.yaml --dry-run  "
        "[dim]# dry-run inside the image[/dim]"
    )
    rprint(
        f"  [dim]6.[/dim] vkit docker run --tag {tag} pipeline.yaml            "
        "[dim]# do the work[/dim]"
    )
    rprint(
        "\n[dim]Tip:[/dim] `vkit operators search <keyword>` discovers built-ins; "
        "`vkit datasets --recipe-only` lists ingest-ready public datasets."
    )


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
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Skip static field-contract pre-flight checks."
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        help=(
            "Check the pipeline's operators fit this image tag "
            "(slim|asr|diarize|tts|fish-speech|latest). "
            "Catches wrong-image and stale-image mismatches before a run."
        ),
    ),
) -> None:
    from voxkitchen.cli.validate import validate_command

    validate_command(pipeline, preflight=not no_preflight, tag=tag)


@app.command(help="Pretty-print a pipeline YAML with each stage's field contracts.")
def show(
    pipeline: Path = typer.Argument(..., help="Pipeline YAML path."),
) -> None:
    from voxkitchen.cli.show_cmd import show_command

    show_command(pipeline)


@app.command(
    help=(
        "Execute a pipeline in the current environment. "
        "Container entrypoint; host users should prefer `vkit docker run`."
    )
)
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
    no_preflight: bool = typer.Option(
        False, "--no-preflight", help="Skip static field-contract pre-flight checks."
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
        no_preflight=no_preflight,
    )


@app.command(help="Download a dataset.")
def download(
    recipe: str = typer.Argument(..., help="Recipe name (e.g., librispeech, aishell, fleurs)"),
    root: Path = typer.Option(..., "--root", help="Directory to download into"),
    subsets: str | None = typer.Option(None, "--subsets", help="Comma-separated subset names"),
) -> None:
    from voxkitchen.cli.download_cmd import download_command

    try:
        download_command(recipe_name=recipe, root=root, subsets=subsets)
    except Exception as exc:
        rprint(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command(help="Launch local Gradio panel to explore a CutSet.")
def viz(
    path: Path = typer.Argument(..., help="CutSet manifest path."),
    port: int = typer.Option(7860, "--port", help="Port for local server."),
) -> None:
    try:
        from voxkitchen.viz.panel.app import launch

        launch(str(path), port=port)
    except ImportError:
        rprint(
            "[red]error:[/red] Gradio panel dependencies are not installed "
            "in this environment. Use `vkit inspect cuts` for CLI inspection."
        )
        raise typer.Exit(code=1) from None


@app.command(help="Generate a shareable HTML dataset card from a CutSet manifest.")
def card(
    manifest: Path = typer.Argument(..., help="Path to a cuts.jsonl.gz manifest."),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Output HTML path (default: dataset_card.html)."
    ),
    title: str = typer.Option("", "--title", help="Card title."),
    description: str = typer.Option("", "--description", help="Short dataset description."),
    catalog_id: str | None = typer.Option(
        None,
        "--catalog-id",
        help="Pre-fill title/description and a Source section from "
        "voxkitchen/datasets/catalog.yaml (e.g. --catalog-id librispeech).",
    ),
) -> None:
    from voxkitchen.cli.card_cmd import card_command

    card_command(
        manifest,
        out=out,
        title=title,
        description=description,
        catalog_id=catalog_id,
    )


if __name__ == "__main__":
    app()
