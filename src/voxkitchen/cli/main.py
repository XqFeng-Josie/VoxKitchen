"""Top-level Typer application exposing the `vkit` CLI.

Plan 1 ships a placeholder skeleton: each subcommand prints a
"not yet implemented" message. Later plans replace these stubs with
real behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import typer
from rich import print as rprint

app = typer.Typer(
    name="vkit",
    help="VoxKitchen — declarative speech data processing toolkit.",
    no_args_is_help=True,
    add_completion=False,
)


def _not_implemented(command: str) -> NoReturn:
    rprint(f"[yellow]vkit {command}[/yellow]: not yet implemented in this build.")
    raise typer.Exit(code=1)


@app.command(help="Scaffold a new pipeline project directory.")
def init(path: str = typer.Argument(..., help="Target directory.")) -> None:
    _not_implemented(f"init {path}")


@app.command(help="Build an initial CutSet from a data source.")
def ingest(
    source: str = typer.Option(..., "--source", help="dir | manifest | recipe"),
    out: str = typer.Option(..., "--out", help="Output cuts.jsonl.gz path"),
) -> None:
    _not_implemented(f"ingest --source {source} --out {out}")


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
        None, "--resume-from", help="Stage to resume from (not yet implemented)."
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


@app.command(help="Inspect cuts, recordings, run progress, trace, or errors.")
def inspect(
    subcommand: str = typer.Argument(..., help="cuts | recordings | run | trace | errors"),
    path: str = typer.Argument(..., help="Target path."),
) -> None:
    # TODO: Replace with a Typer sub-app (app.add_typer) when cli/inspect.py lands.
    # The real signatures differ structurally — e.g. `inspect trace <cut_id> --in <path>`
    # cannot be represented by the current flat (subcommand, path) placeholder.
    _not_implemented(f"inspect {subcommand} {path}")


@app.command(help="Launch local Gradio panel to explore a CutSet.")
def viz(path: str = typer.Argument(..., help="CutSet or work_dir path.")) -> None:
    _not_implemented(f"viz {path}")


if __name__ == "__main__":
    app()
