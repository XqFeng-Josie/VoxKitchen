"""Implementation of `vkit recipes` command."""

from __future__ import annotations

from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _download_source_label(download_urls: dict[str, list[str]]) -> str:
    """Pick a short, accurate source label from a recipe's download URLs.

    Inspects the host of the first URL: openslr.org → ``openslr``,
    keithito.com / data.keithito.com → ``keithito``, anything else falls
    back to the bare hostname. This keeps `vkit recipes` honest when a
    new recipe pulls from a non-OpenSLR mirror.
    """
    for urls in download_urls.values():
        if not urls:
            continue
        host = urlparse(urls[0]).hostname or ""
        host = host.lower()
        if "openslr" in host:
            return "openslr"
        if "keithito" in host:
            return "keithito"
        if "huggingface" in host or "hf.co" in host:
            return "HuggingFace"
        # Strip a leading "www." for a tidier label.
        return host.removeprefix("www.")
    return "url"


recipes_app = typer.Typer(
    name="recipes",
    help="List available dataset recipes.",
    invoke_without_command=True,
)


@recipes_app.callback()
def list_recipes(ctx: typer.Context) -> None:
    """List all available dataset recipes and their download support."""
    if ctx.invoked_subcommand is not None:
        return

    from voxkitchen.ingest.recipes import _RECIPES

    t = Table(title=f"Available recipes ({len(_RECIPES)})")
    t.add_column("Recipe", style="bold")
    t.add_column("Download")
    t.add_column("Description")

    for name in sorted(_RECIPES.keys()):
        recipe = _RECIPES[name]
        if recipe.download_urls:
            dl = f"[green]{_download_source_label(recipe.download_urls)}[/green]"
        elif (
            hasattr(recipe, "download")
            and type(recipe).download is not type(recipe).__mro__[1].download  # type: ignore[attr-defined]
        ):
            # Has custom download() override (e.g. FLEURS via HuggingFace)
            dl = "[green]HuggingFace[/green]"
        else:
            dl = "[dim]manual[/dim]"

        doc = type(recipe).__doc__ or ""
        first_line = doc.strip().split("\n")[0] if doc.strip() else ""
        # Fallback description from module docstring
        if not first_line:
            mod = type(recipe).__module__
            import importlib

            m = importlib.import_module(mod)
            first_line = (m.__doc__ or "").strip().split("\n")[0]

        t.add_row(name, dl, first_line)

    console.print(t)
    console.print()
    console.print(
        "[dim]Download:[/dim] [bold]vkit docker download --tag slim <recipe> --root ./data/<recipe>[/bold]"
    )
    console.print(
        "[dim]Use in pipeline:[/dim] ingest.source=recipe, recipe=<name>, args.root=./data/<recipe>"
    )
    console.print()
