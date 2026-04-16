"""Provenance record: describes how a Cut was produced."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Provenance(BaseModel):
    """Where this Cut came from in the pipeline.

    - ``source_cut_id`` is ``None`` only for freshly-ingested Cuts that have no
      parent in the VoxKitchen data model. Every derived Cut (VAD output,
      filtered subset, etc.) points back to its parent.
    - ``generated_by`` is an operator identifier, conventionally
      ``<operator_name>@<version>`` (e.g. ``"silero_vad@0.4.1"``).
    - ``stage_name`` matches the stage name in the pipeline YAML, so
      ``vkit inspect`` can cross-reference with the run's ``run.yaml``.
    """

    model_config = ConfigDict(extra="forbid")

    source_cut_id: str | None
    generated_by: str
    stage_name: str
    created_at: datetime
    pipeline_run_id: str
