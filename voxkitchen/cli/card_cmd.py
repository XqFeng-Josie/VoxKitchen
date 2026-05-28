"""Implementation of the `vkit card` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from voxkitchen.schema.cutset import CutSet


def card_command(
    manifest: Path,
    out: Path | None = None,
    title: str = "",
    description: str = "",
) -> None:
    """Generate a shareable HTML dataset card from a CutSet manifest."""
    try:
        # jinja2 (the 'viz' extra) is imported lazily inside the generator, so
        # check it here to give a friendly message instead of a raw traceback.
        import jinja2  # noqa: F401
    except ImportError as exc:
        rprint(
            r"[red]error:[/red] the dataset card needs the 'viz' extra. "
            r"Install it with `pip install voxkitchen\[viz]`."
        )
        raise typer.Exit(code=1) from exc

    from voxkitchen.viz.card.generator import generate_dataset_card

    cuts = CutSet.from_jsonl_gz(manifest)
    target = out or Path("dataset_card.html")
    generate_dataset_card(cuts, target, title=title, description=description)
    rprint(f"[green]wrote dataset card:[/green] {target}")
