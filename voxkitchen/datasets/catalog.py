"""Dataset catalog: structured metadata for the decision-support catalog.

A single ``catalog.yaml`` (one ``DatasetEntry`` per dataset) is the authoritative
source; ``voxkitchen.datasets.catalog_gen`` renders it into ``docs/datasets/``.
The catalog is informational — entries center on a curated ``recommendation``
and an ``homepage`` access link; ``recipe`` (downloadable via VoxKitchen) and
``recommended_pipeline`` are optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

Task = Literal["asr", "tts", "speaker", "multilingual", "augmentation"]


class DatasetEntry(BaseModel):
    """One catalog entry. Extra fields are rejected to catch typos in catalog.yaml."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    task: list[Task]
    languages: list[str]
    license: str
    summary: str
    homepage: str
    recommendation: str
    hours: float | None = None
    domain: str | None = None
    recipe: str | None = None
    recommended_pipeline: str | None = None
    paper: str | None = None
    notes: str | None = None


# Repo root = three levels up from this file (voxkitchen/datasets/catalog.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"


class CatalogError(ValueError):
    """A catalog.yaml entry is malformed or references something that doesn't exist."""


def load_catalog(path: Path | None = None) -> list[DatasetEntry]:
    """Parse + validate catalog.yaml. Raises CatalogError on any problem."""
    path = path or DEFAULT_CATALOG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_entries = raw.get("entries", [])
    if not isinstance(raw_entries, list):
        raise CatalogError("catalog.yaml must have a top-level 'entries:' list")

    entries: list[DatasetEntry] = []
    for i, item in enumerate(raw_entries):
        try:
            entries.append(DatasetEntry(**item))
        except Exception as exc:  # pydantic ValidationError or bad shape
            raise CatalogError(f"entry #{i} invalid: {exc}") from exc

    _cross_validate(entries)
    return entries


def _cross_validate(entries: list[DatasetEntry]) -> None:
    from voxkitchen.ingest.recipes import get_recipe

    seen: set[str] = set()
    for e in entries:
        if e.id in seen:
            raise CatalogError(f"duplicate id {e.id!r}")
        seen.add(e.id)
        if e.recipe is not None:
            try:
                get_recipe(e.recipe)  # public API; raises KeyError if unregistered
            except KeyError as exc:
                raise CatalogError(
                    f"entry {e.id!r}: recipe {e.recipe!r} is not a registered recipe"
                ) from exc
        if e.recommended_pipeline is not None:
            if not (_REPO_ROOT / e.recommended_pipeline).is_file():
                raise CatalogError(
                    f"entry {e.id!r}: recommended_pipeline "
                    f"{e.recommended_pipeline!r} does not exist"
                )
