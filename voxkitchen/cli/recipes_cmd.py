"""Implementation of `vkit recipes` command."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from voxkitchen.datasets.recipe_meta import download_source_label as _plain_source_label
from voxkitchen.datasets.recipe_meta import format_size_range as _plain_size_range

console = Console()


def _format_size_column(download_sizes: dict[str, int]) -> str:
    """Render the Size column entry for one recipe with Rich markup.

    Delegates plain formatting to :func:`voxkitchen.datasets.recipe_meta.format_size_range`
    and wraps the empty/unknown case in ``[dim]``…``[/dim]``.

    - Empty dict (manual / HuggingFace-streaming recipes) → "[dim]-[/dim]".
    - Single-subset dict → exact size (e.g. "11.0 GB").
    - Multi-subset dict → range across subsets ("330 MB - 28.5 GB").
    """
    plain = _plain_size_range(download_sizes)
    return "[dim]-[/dim]" if plain == "—" else plain


def _download_source_label(download_urls: dict[str, list[str]]) -> str:
    """Pick a short, accurate source label from a recipe's download URLs.

    Delegates to :func:`voxkitchen.datasets.recipe_meta.download_source_label`.
    The caller (``list_recipes``) wraps the result in Rich colour markup.
    """
    return _plain_source_label(download_urls)


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
    t.add_column("Size", justify="right")
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

        size_label = _format_size_column(recipe.download_sizes)

        doc = type(recipe).__doc__ or ""
        first_line = doc.strip().split("\n")[0] if doc.strip() else ""
        # Fallback description from module docstring
        if not first_line:
            mod = type(recipe).__module__
            import importlib

            m = importlib.import_module(mod)
            first_line = (m.__doc__ or "").strip().split("\n")[0]

        t.add_row(name, dl, size_label, first_line)

    console.print(t)
    console.print()
    console.print(
        "[dim]Download:[/dim] [bold]vkit docker download --tag slim <recipe> --root ./data/<recipe>[/bold]"
    )
    console.print(
        "[dim]Use in pipeline:[/dim] ingest.source=recipe, recipe=<name>, args.root=./data/<recipe>"
    )
    console.print()
