"""Implementation of ``vkit download`` command."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint

from voxkitchen.cli.hints import warn_if_unmanaged_runtime
from voxkitchen.ingest.recipes import get_recipe


def download_command(
    recipe_name: str,
    root: Path,
    subsets: str | None = None,
) -> None:
    """Download a dataset using its recipe."""
    # Recipes that download from the network rely on packages (e.g.
    # `datasets` for HuggingFace-hosted corpora) that ship in the Docker
    # images, not in the lightweight PyPI launcher. Warn host users to
    # take the supported `vkit docker download` path.
    warn_if_unmanaged_runtime(
        command="download",
        recommended="vkit docker download <recipe>",
    )
    recipe = get_recipe(recipe_name)

    subset_list = [s.strip() for s in subsets.split(",")] if subsets else None

    rprint(f"[bold]Downloading {recipe_name}[/bold] → {root}")
    if subset_list:
        rprint(f"  Subsets: {subset_list}")

    recipe.download(root, subset_list)

    rprint("[green]Download complete.[/green]")
    rprint("\nUse in pipeline YAML:")
    rprint("  ingest:")
    rprint("    source: recipe")
    rprint(f"    recipe: {recipe_name}")
    rprint("    args:")
    rprint(f"      root: {root}")
