"""Implementation of ``vkit datasets`` — terminal browser for the catalog.

The catalog ships with the wheel as ``voxkitchen/datasets/catalog.yaml`` —
this command loads it and presents three views without opening the docs
site:

  - ``vkit datasets`` / ``vkit datasets list``: a filterable table
  - ``vkit datasets show <id>``: full detail panel for one entry
  - ``vkit datasets search <query>``: substring match across id/name/summary

Filters (``--task`` / ``--language`` / ``--recipe-only``) compose. All output
is Rich-rendered; nothing is downloaded — this is a navigation aid, not an
ingest path.
"""

from __future__ import annotations

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from voxkitchen.datasets.catalog import DatasetEntry, load_catalog

console = Console()


datasets_app = typer.Typer(
    name="datasets",
    help="Browse the VoxKitchen dataset catalog.",
    invoke_without_command=True,
    no_args_is_help=False,
)


def _hours_cell(e: DatasetEntry) -> str:
    """Render the Hours column — short ``g``-format or ``-`` for unknown."""
    if e.hours is None:
        return "[dim]-[/dim]"
    # 5.2 → "5.2", 1200.0 → "1200" — ``g`` drops trailing zeros.
    return f"{e.hours:g}"


def _access_cell(e: DatasetEntry) -> str:
    """``recipe-name`` in green when downloadable, ``manual`` in dim otherwise."""
    if e.recipe:
        return f"[green]{e.recipe}[/green]"
    return "[dim]manual[/dim]"


def _filter_entries(
    entries: list[DatasetEntry],
    *,
    task: str | None,
    language: str | None,
    recipe_only: bool,
    query: str | None,
) -> list[DatasetEntry]:
    """Apply CLI filters in a deterministic order (matches docs site).

    All filters are case-insensitive; ``query`` matches against id, name,
    and summary so users can search by either a short slug or natural text.
    """
    result = entries
    if task:
        t = task.lower()
        result = [e for e in result if t in e.task]
    if language:
        lng = language.lower()
        result = [e for e in result if lng in [x.lower() for x in e.languages]]
    if recipe_only:
        result = [e for e in result if e.recipe]
    if query:
        q = query.lower()
        result = [
            e for e in result if q in e.id.lower() or q in e.name.lower() or q in e.summary.lower()
        ]
    return result


@datasets_app.callback()
def list_datasets(
    ctx: typer.Context,
    task: str | None = typer.Option(
        None,
        "--task",
        "-t",
        help="Filter by task tag (asr/tts/speaker/multilingual/emotion/augmentation).",
    ),
    language: str | None = typer.Option(
        None, "--language", "-l", help="Filter by ISO language code (en/zh/multi/ja/...)."
    ),
    recipe_only: bool = typer.Option(
        False,
        "--recipe-only",
        help="Only show entries that are downloadable via `vkit docker download`.",
    ),
    query: str | None = typer.Option(
        None, "--query", "-q", help="Substring match against id / name / summary."
    ),
) -> None:
    """List catalog entries as a table. Default view when no subcommand is given."""
    if ctx.invoked_subcommand is not None:
        return

    entries = load_catalog()
    entries = _filter_entries(
        entries, task=task, language=language, recipe_only=recipe_only, query=query
    )

    if not entries:
        rprint("[yellow]no entries match those filters.[/yellow]")
        rprint("[dim]hint: try `vkit datasets` with no filters to see all 60 entries.[/dim]")
        raise typer.Exit(0)

    t = Table(title=f"VoxKitchen dataset catalog ({len(entries)} entries)")
    t.add_column("ID", style="bold cyan")
    t.add_column("Name", overflow="fold")
    t.add_column("Task")
    t.add_column("Lang")
    t.add_column("Hours", justify="right")
    t.add_column("License", overflow="fold")
    t.add_column("Access")

    for e in entries:
        t.add_row(
            e.id,
            e.name,
            ", ".join(e.task),
            ", ".join(e.languages),
            _hours_cell(e),
            e.license,
            _access_cell(e),
        )
    console.print(t)


@datasets_app.command("show")
def show(
    dataset_id: str = typer.Argument(..., help="Catalog ID (e.g. librispeech)."),
) -> None:
    """Print the full record for one catalog entry."""
    entries = load_catalog()
    match = next((e for e in entries if e.id == dataset_id), None)
    if match is None:
        rprint(f"[red]error:[/red] no catalog entry with id {dataset_id!r}.")
        rprint("[dim]hint: `vkit datasets search <substring>` to find the right id.[/dim]")
        raise typer.Exit(1)

    body_lines: list[str] = []
    body_lines.append(f"[bold]Task:[/bold]      {', '.join(match.task)}")
    body_lines.append(f"[bold]Languages:[/bold] {', '.join(match.languages)}")
    if match.hours is not None:
        body_lines.append(f"[bold]Hours:[/bold]     {match.hours:g}")
    if match.domain:
        body_lines.append(f"[bold]Domain:[/bold]    {match.domain}")
    body_lines.append(f"[bold]License:[/bold]   {match.license}")
    body_lines.append(f"[bold]Homepage:[/bold]  {match.homepage}")
    if match.paper:
        body_lines.append(f"[bold]Paper:[/bold]     {match.paper}")
    if match.recipe:
        body_lines.append(
            f"[bold]Recipe:[/bold]    [green]{match.recipe}[/green] "
            f"[dim](run `vkit docker download {match.recipe} --root <dir>`)[/dim]"
        )
    if match.recommended_pipeline:
        body_lines.append(f"[bold]Pipeline:[/bold]  {match.recommended_pipeline}")
    body_lines.append("")
    body_lines.append("[bold]Summary[/bold]")
    body_lines.append(match.summary)
    body_lines.append("")
    body_lines.append("[bold]Recommendation[/bold]")
    body_lines.append(match.recommendation)
    if match.notes:
        body_lines.append("")
        body_lines.append("[bold]Notes[/bold]")
        body_lines.append(match.notes)

    console.print(Panel("\n".join(body_lines), title=f"[bold]{match.name}[/bold] ({match.id})"))


@datasets_app.command("search")
def search(
    query: str = typer.Argument(..., help="Substring (case-insensitive) to look for."),
    task: str | None = typer.Option(None, "--task", "-t"),
    language: str | None = typer.Option(None, "--language", "-l"),
    recipe_only: bool = typer.Option(False, "--recipe-only"),
) -> None:
    """Substring search across id, name, and summary; same table as list."""
    entries = load_catalog()
    entries = _filter_entries(
        entries, task=task, language=language, recipe_only=recipe_only, query=query
    )

    if not entries:
        rprint(f"[yellow]no entries match {query!r}.[/yellow]")
        raise typer.Exit(0)

    t = Table(title=f"Search: {query!r} → {len(entries)} match(es)")
    t.add_column("ID", style="bold cyan")
    t.add_column("Name", overflow="fold")
    t.add_column("Task")
    t.add_column("Lang")
    t.add_column("Hours", justify="right")
    t.add_column("Access")

    for e in entries:
        t.add_row(
            e.id,
            e.name,
            ", ".join(e.task),
            ", ".join(e.languages),
            _hours_cell(e),
            _access_cell(e),
        )
    console.print(t)
