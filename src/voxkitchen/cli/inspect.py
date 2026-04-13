"""Typer sub-app for `vkit inspect` subcommands."""

from __future__ import annotations

from pathlib import Path

import typer

inspect_app = typer.Typer(
    name="inspect",
    help="Inspect cuts, recordings, run progress, trace, or errors.",
)


@inspect_app.command()
def cuts(path: Path = typer.Argument(..., help="Path to cuts.jsonl.gz")) -> None:
    """Show statistics for a CutSet manifest."""
    from voxkitchen.viz.cli import render_cuts_stats

    render_cuts_stats(path)


@inspect_app.command(name="run")
def run_summary(
    work_dir: Path = typer.Argument(..., help="Pipeline work directory"),
) -> None:
    """Show pipeline run stage summary."""
    from voxkitchen.viz.cli import render_run_summary

    render_run_summary(work_dir)


@inspect_app.command()
def errors(
    work_dir: Path = typer.Argument(..., help="Pipeline work directory"),
) -> None:
    """Show errors from pipeline stage logs."""
    from voxkitchen.viz.cli import render_errors

    render_errors(work_dir)
