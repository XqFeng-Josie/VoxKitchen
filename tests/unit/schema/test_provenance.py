"""Unit tests for voxkitchen.schema.provenance.Provenance."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from voxkitchen.schema.provenance import Provenance


def test_provenance_minimal_construction() -> None:
    p = Provenance(
        source_cut_id="cut-parent",
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    assert p.source_cut_id == "cut-parent"
    assert p.generated_by == "silero_vad@0.4.1"
    assert p.stage_name == "02_vad"
    assert p.pipeline_run_id == "run-a1b2c3"


def test_provenance_source_cut_id_can_be_none_for_root_cuts() -> None:
    p = Provenance(
        source_cut_id=None,
        generated_by="dir_scan",
        stage_name="00_ingest",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    assert p.source_cut_id is None


def test_provenance_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        Provenance(  # type: ignore[call-arg]
            generated_by="silero_vad",
            stage_name="02_vad",
            created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
            pipeline_run_id="run-a1b2c3",
        )


def test_provenance_round_trips_through_json() -> None:
    original = Provenance(
        source_cut_id="cut-parent",
        generated_by="silero_vad@0.4.1",
        stage_name="02_vad",
        created_at=datetime(2026, 4, 11, 10, 30, 0, tzinfo=timezone.utc),
        pipeline_run_id="run-a1b2c3",
    )
    blob = original.model_dump_json()
    restored = Provenance.model_validate_json(blob)
    assert restored == original


def test_provenance_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Provenance.model_validate(
            {
                "source_cut_id": "cut-parent",
                "generated_by": "silero_vad",
                "stage_name": "02_vad",
                "created_at": "2026-04-11T10:30:00Z",
                "pipeline_run_id": "run-a1b2c3",
                "surprise_field": "boom",
            }
        )
