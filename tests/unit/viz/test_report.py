"""Unit tests for voxkitchen.viz.report.generator."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from voxkitchen.schema.cut import Cut
from voxkitchen.schema.cutset import CutSet
from voxkitchen.schema.io import SCHEMA_VERSION, HeaderRecord
from voxkitchen.schema.provenance import Provenance
from voxkitchen.viz.report.generator import generate_report


def _prov() -> Provenance:
    return Provenance(
        source_cut_id=None,
        generated_by="test",
        stage_name="test",
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
    )


def _cut(cid: str, duration: float = 1.0) -> Cut:
    return Cut(
        id=cid,
        recording_id="rec",
        start=0.0,
        duration=duration,
        supervisions=[],
        provenance=_prov(),
    )


def _write_stage(work_dir: Path, stage_name: str, cuts: list[Cut], success: bool = True) -> None:
    stage_dir = work_dir / stage_name
    stage_dir.mkdir(parents=True)
    header = HeaderRecord(
        schema_version=SCHEMA_VERSION,
        created_at=datetime(2026, 4, 11, tzinfo=timezone.utc),
        pipeline_run_id="run",
        stage_name=stage_name,
    )
    CutSet(cuts).to_jsonl_gz(stage_dir / "cuts.jsonl.gz", header)
    if success:
        (stage_dir / "_SUCCESS").touch()


def test_generate_report_creates_html(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_stage(work_dir, "00_vad", [_cut("c0", 2.0), _cut("c1", 3.0)])

    report_path = generate_report(work_dir)

    assert report_path.exists()
    assert report_path == work_dir / "report.html"
    html = report_path.read_text()
    assert "VoxKitchen" in html
    assert "2 cuts" in html or "<b>2</b>" in html


def test_report_contains_stage_info(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_stage(work_dir, "00_vad", [_cut("c0"), _cut("c1")])
    _write_stage(work_dir, "01_pack", [_cut("c0"), _cut("c1")])

    html = generate_report(work_dir).read_text()

    assert "00_vad" in html
    assert "01_pack" in html


def test_report_without_plotly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report must still generate when plotly is not installed."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_stage(work_dir, "00_vad", [_cut("c0", 1.5)])

    # Make plotly appear unimportable
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "plotly.io", None)

    report_path = generate_report(work_dir)

    assert report_path.exists()
    html = report_path.read_text()
    assert "VoxKitchen" in html
    assert "chart unavailable" in html
