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
    catalog_id: str | None = None,
) -> None:
    """Generate a shareable HTML dataset card from a CutSet manifest.

    When ``catalog_id`` is set, pre-fill title/description and a "Source" block
    from the matching entry in ``voxkitchen/datasets/catalog.yaml``. Explicit
    ``title`` / ``description`` flags still override the catalog values.
    """
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

    source: dict[str, str] | None = None
    if catalog_id:
        from voxkitchen.datasets.catalog import load_catalog

        entries = {e.id: e for e in load_catalog()}
        if catalog_id not in entries:
            rprint(
                f"[red]error:[/red] catalog id {catalog_id!r} not found. "
                "Run `vkit recipes` or browse docs/datasets/ for valid ids."
            )
            raise typer.Exit(code=1)
        entry = entries[catalog_id]
        title = title or entry.name
        description = description or entry.summary
        source = {
            "license": entry.license,
            "homepage": entry.homepage,
            "paper": entry.paper or "",
            "recommendation": entry.recommendation,
        }

    from voxkitchen.viz.card.generator import generate_dataset_card

    cuts = CutSet.from_jsonl_gz(manifest)
    target = out or Path("dataset_card.html")
    generate_dataset_card(cuts, target, title=title, description=description, source=source)
    rprint(f"[green]wrote dataset card:[/green] {target}")
