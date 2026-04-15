"""Implementation of ``vkit download`` command."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint

from voxkitchen.ingest.recipes import get_recipe


def download_command(
    recipe_name: str,
    root: Path,
    subsets: str | None = None,
) -> None:
    """Download a dataset using its recipe."""
    recipe = get_recipe(recipe_name)

    subset_list = [s.strip() for s in subsets.split(",")] if subsets else None

    rprint(f"[bold]Downloading {recipe_name}[/bold] → {root}")
    if subset_list:
        rprint(f"  Subsets: {subset_list}")

    recipe.download(root, subset_list)

    rprint(f"[green]Download complete.[/green]")
    rprint(f"\nUse in pipeline YAML:")
    rprint(f"  ingest:")
    rprint(f"    source: recipe")
    rprint(f"    recipe: {recipe_name}")
    rprint(f"    args:")
    rprint(f"      root: {root}")
