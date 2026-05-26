"""Dataset catalog: structured metadata for the decision-support catalog.

A single ``catalog.yaml`` (one ``DatasetEntry`` per dataset) is the authoritative
source; ``voxkitchen.datasets.catalog_gen`` renders it into ``docs/datasets/``.
The catalog is informational — entries center on a curated ``recommendation``
and an ``homepage`` access link; ``recipe`` (downloadable via VoxKitchen) and
``recommended_pipeline`` are optional.
"""

from __future__ import annotations

from typing import Literal

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
