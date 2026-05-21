"""Typer sub-app for `vkit inspect` subcommands.

Each subcommand delegates to a ``render_*`` helper in
:mod:`voxkitchen.viz.cli`. The helpers return ``False`` when the requested
input is missing or unreadable; we convert that into ``typer.Exit(code=1)``
so shell scripts that pipe ``vkit inspect …`` see a real non-zero exit
on failure.
"""

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

    if not render_cuts_stats(path):
        raise typer.Exit(code=1)


@inspect_app.command(name="run")
def run_summary(
    work_dir: Path = typer.Argument(..., help="Pipeline work directory"),
) -> None:
    """Show pipeline run stage summary."""
    from voxkitchen.viz.cli import render_run_summary

    if not render_run_summary(work_dir):
        raise typer.Exit(code=1)


@inspect_app.command()
def trace(
    cut_id: str = typer.Argument(..., help="Cut ID to trace"),
    work_dir: Path = typer.Option(..., "--in", help="Pipeline work directory"),
) -> None:
    """Trace the provenance chain for a cut across pipeline stages."""
    from voxkitchen.viz.cli import render_trace

    if not render_trace(cut_id, work_dir):
        raise typer.Exit(code=1)


@inspect_app.command()
def errors(
    work_dir: Path = typer.Argument(..., help="Pipeline work directory"),
) -> None:
    """Show errors from pipeline stage logs."""
    from voxkitchen.viz.cli import render_errors

    if not render_errors(work_dir):
        raise typer.Exit(code=1)
