"""Cut: a trainable sample referencing a Recording slice with Supervisions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from voxkitchen.schema.provenance import Provenance
from voxkitchen.schema.recording import Recording
from voxkitchen.schema.supervision import Supervision


class Cut(BaseModel):
    """A trainable sample — the unit flowing through a pipeline.

    A Cut references a ``[start, start+duration)`` slice of a Recording plus
    the Supervisions that fall within it. Operators transform CutSets by
    producing new Cuts (segmentation creates children, quality filters drop
    Cuts, ASR appends Supervisions) — original Cuts are never mutated.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    recording_id: str
    start: float
    duration: float
    channel: int | list[int] | None = None
    recording: Recording | None = None

    supervisions: list[Supervision]
    metrics: dict[str, float] = {}
    provenance: Provenance
    custom: dict[str, Any] = {}
