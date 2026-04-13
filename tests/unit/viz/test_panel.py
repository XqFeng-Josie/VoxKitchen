"""Tests for Gradio panel (skipped if gradio not installed)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")

from voxkitchen.schema.cut import Cut  # noqa: E402
from voxkitchen.schema.cutset import CutSet  # noqa: E402
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord  # noqa: E402
from voxkitchen.schema.provenance import Provenance  # noqa: E402
from voxkitchen.viz.panel.app import create_app  # noqa: E402


def _write_manifest(path: Path, n: int = 3) -> None:
    cuts = [
        Cut(
            id=f"c{i}",
            recording_id="rec",
            start=0,
            duration=float(i + 1),
            supervisions=[],
            provenance=Provenance(
                source_cut_id=None,
                generated_by="test",
                stage_name="test",
                created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
                pipeline_run_id="run",
            ),
        )
        for i in range(n)
    ]
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
        stage_name="test",
    )
    CutSet(cuts).to_jsonl_gz(path, header)


def test_create_app_returns_blocks(tmp_path: Path) -> None:
    manifest = tmp_path / "cuts.jsonl.gz"
    _write_manifest(manifest)
    app = create_app(str(manifest))
    assert isinstance(app, gr.Blocks)


def test_create_app_with_empty_cutset(tmp_path: Path) -> None:
    manifest = tmp_path / "cuts.jsonl.gz"
    _write_manifest(manifest, n=0)
    app = create_app(str(manifest))
    assert isinstance(app, gr.Blocks)
